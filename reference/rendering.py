"""Process-local EGL lifecycle repair, applied equally to every evaluation arm."""
_installed=False
EVENTS={'safe_context_frees':0,'render_calls':0,'read_calls':0,'observation_checks':0,'errors':[]}


def install():
    global _installed
    if _installed:return
    from robosuite.utils import binding_utils as b
    from robosuite.renderers.context import egl_context as ec
    from OpenGL import EGL,GL
    import numpy as np
    import mujoco
    original_make_current=ec.EGLGLContext.make_current
    def make_current(self):
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)
        return original_make_current(self)
    ec.EGLGLContext.make_current=make_current
    def safe_destructor(self):
        if getattr(self,'_repro_freed',False):return
        self._repro_freed=True
        gl_ctx=getattr(self,'gl_ctx',None);con=getattr(self,'con',None)
        if gl_ctx is None:return
        previous=EGL.eglGetCurrentContext();display=EGL.eglGetCurrentDisplay()
        draw=EGL.eglGetCurrentSurface(EGL.EGL_DRAW);read=EGL.eglGetCurrentSurface(EGL.EGL_READ)
        own=getattr(gl_ctx,'_context',None)
        # PyOpenGL's .address dereferences NULL opaque pointers. No active
        # context is a normal teardown state; compare pointer values instead.
        import ctypes
        previous_address=ctypes.cast(previous,ctypes.c_void_p).value if previous else None
        own_address=ctypes.cast(own,ctypes.c_void_p).value if own else None
        try:
            gl_ctx.make_current()
            if con is not None:con.free()
            gl_ctx.free()
            EVENTS['safe_context_frees']+=1
        except Exception as exc:
            EVENTS['errors'].append('context_free: '+repr(exc))
        finally:
            if previous and previous_address!=own_address:
                try:
                    EGL.eglBindAPI(EGL.EGL_OPENGL_API)
                    if not EGL.eglMakeCurrent(display,draw,read,previous):
                        EVENTS['errors'].append('Could not restore active EGL context')
                except Exception as exc:EVENTS['errors'].append('context_restore: '+repr(exc))
            self.con=None;self.gl_ctx=None
    b.MjRenderContext.__del__=safe_destructor
    original_render=b.MjRenderContext.render
    original_read=b.MjRenderContext.read_pixels
    def render(self,*args,**kwargs):
        self.gl_ctx.make_current();EVENTS['render_calls']+=1
        return original_render(self,*args,**kwargs)
    def read_pixels(self,width,height,depth=False,segmentation=False):
        self.gl_ctx.make_current();EVENTS['read_calls']+=1
        if depth or segmentation:return original_read(self,width,height,depth=depth,segmentation=segmentation)
        # Unwritten np.empty buffers previously looked like valid finite RGB.
        # A sentinel plus GL error check makes a failed read explicit.
        while GL.glGetError()!=GL.GL_NO_ERROR:pass
        rgb=np.empty((height,width,3),dtype=np.uint8);rgb[:]=[255,0,255]
        viewport=mujoco.MjrRect(0,0,width,height)
        mujoco.mjr_readPixels(rgb=rgb,depth=None,viewport=viewport,con=self.con)
        error=GL.glGetError()
        if error!=GL.GL_NO_ERROR or np.all(rgb==np.array([255,0,255],dtype=np.uint8)):
            raise RuntimeError('Invalid camera readback: GL error or untouched RGB sentinel')
        if EVENTS['errors']:raise RuntimeError('EGL lifecycle error: '+repr(EVENTS['errors']))
        return rgb
    b.MjRenderContext.render=render;b.MjRenderContext.read_pixels=read_pixels
    _installed=True


def health_metrics(image):
    """Report spatial structure; this never modifies or repairs an observation."""
    import numpy as np
    a=np.asarray(image,dtype=np.float32)
    if a.shape[0]==3:a=a.transpose(1,2,0)
    if float(a.max())>1.5:a=a/255.
    var=float(np.mean((a-a.mean(axis=(0,1),keepdims=True))**2))
    edge=float(.5*(np.mean(np.diff(a,axis=0)**2)+np.mean(np.diff(a,axis=1)**2)))
    flat=float(.5*(np.mean(np.max(np.abs(np.diff(a,axis=0)),axis=-1)<2/255.)+
                     np.mean(np.max(np.abs(np.diff(a,axis=1)),axis=-1)<2/255.)))
    return dict(mean=float(a.mean()),std=var**.5,edge_to_variance=edge/max(var,1e-8),flat_fraction=flat)


def check_observation(observation):
    import numpy as np
    cameras=('robot0_agentview_left_image','robot0_agentview_right_image','robot0_eye_in_hand_image')
    for key in cameras:
        a=np.asarray(observation[key])
        if a.shape!=(3,128,128) or not np.isfinite(a).all() or a.min()<-1e-6 or a.max()>1+1e-6:
            raise RuntimeError('Invalid native camera array: '+key)
    values={key:health_metrics(observation[key]) for key in cameras}
    EVENTS['observation_checks']+=1
    # Catastrophic multi-view noise/uniformity, calibrated only on the separate
    # renderer stress test. Abort; never replace cases or alter image inputs.
    if all(v['edge_to_variance']>.8 for v in values.values()) or all(v['std']<1/255. for v in values.values()):
        raise RuntimeError('Catastrophic camera-quality failure: '+repr(values))
    if EVENTS['errors']:raise RuntimeError('Renderer lifecycle failure: '+repr(EVENTS['errors']))
    return values
