"""Three frozen-policy cache controllers. Only public observations are accepted."""
from collections import deque
import numpy as np

METHODS = ('event_refresh', 'anchor_ttl', 'stall_refresh')


def patches(image):
    a = np.asarray(image, dtype=np.float32)
    assert a.shape == (224, 224, 3)
    return a.reshape(16, 14, 16, 14, 3).transpose(0, 2, 1, 3, 4).reshape(256, -1)


class Controller:
    def __init__(self, method):
        assert method in METHODS
        self.method = method
        self.previous_image = None
        self.previous_grip = None
        self.last_grip = None
        self.positions = deque(maxlen=9)
        self.commands = deque(maxlen=8)
        self.grippers = deque(maxlen=9)
        self.cooldown = 0
        self.fresh_remaining = 0
        self.ages = np.zeros(256, dtype=np.int64)
        self.anchor = None
        self.eligible = []
        self.reason = None
        self.interventions = 0

    def before(self, image, eef_position, fingers):
        """Decide whether to drop the cache before this one native model call."""
        a = np.asarray(image, dtype=np.float32) / 255.
        pos = np.asarray(eef_position, dtype=np.float64).copy()
        finger = np.asarray(fingers, dtype=np.float64).copy()
        assert a.shape == (224, 224, 3) and pos.shape == (3,) and finger.shape == (2,)
        assert np.isfinite(a).all() and np.isfinite(pos).all() and np.isfinite(finger).all()
        self.reason = None
        self.positions.append(pos)
        self.grippers.append(finger)
        if self.method == 'event_refresh' and self.previous_image is not None:
            image_delta = float(np.mean(np.abs(a - self.previous_image)))
            grip_flip = self.previous_grip is not None and self.last_grip != self.previous_grip
            if grip_flip or image_delta > .015:
                self.reason = 'gripper_transition' if grip_flip else 'scene_change'
        if self.method == 'stall_refresh':
            if self.cooldown:
                self.cooldown -= 1
            if len(self.positions) == 9 and len(self.commands) == 8 and not self.cooldown:
                motion = float(np.linalg.norm(self.positions[-1] - self.positions[0]))
                finger_motion = float(np.linalg.norm(self.grippers[-1] - self.grippers[0]))
                commanded = float(np.mean([np.linalg.norm(x[:3]) for x in self.commands]))
                if motion < .002 and finger_motion < .003 and commanded > .01:
                    self.fresh_remaining = 3
                    self.cooldown = 16
            if self.fresh_remaining:
                self.reason = 'public_motion_stall'
                self.fresh_remaining -= 1
        self.previous_image = a.copy()
        self.previous_grip = self.last_grip
        self.eligible = []
        if self.reason:
            self.interventions += 1
        return bool(self.reason)

    def filter_indices(self, cropped_image, indices):
        """Conservatively bound each patch's possible reuse age and image drift."""
        if self.method != 'anchor_ttl':
            self.eligible = list(indices)
            return indices
        current = patches(cropped_image)
        if self.anchor is None:
            self.anchor = current.copy()
        numerator = np.sum(current * self.anchor, axis=1)
        similarity = numerator / (np.linalg.norm(current, axis=1) *
                                  np.linalg.norm(self.anchor, axis=1) + 1e-8)
        # IDs include the BOS offset of one. Never add a token the author excluded.
        self.eligible = [int(i) for i in indices
                         if 1 <= i <= 256 and self.ages[i - 1] < 6 and similarity[i - 1] >= .996]
        return self.eligible

    def after(self, action, cropped_image, reused):
        self.commands.append(np.asarray(action, dtype=np.float64).copy())
        self.last_grip = bool(float(action[-1]) >= .5)
        if self.method == 'anchor_ttl':
            current = patches(cropped_image)
            if self.anchor is None:
                self.anchor = current.copy()
            mask = np.zeros(256, dtype=bool)
            if reused:
                mask[np.asarray(self.eligible, dtype=int) - 1] = True
            self.ages[mask] += 1
            self.ages[~mask] = 0
            self.anchor[~mask] = current[~mask]
            assert int(self.ages.max()) <= 6

    def diagnostics(self):
        return {'method': self.method, 'refresh_reason': self.reason,
                'cache_resets': self.interventions, 'eligible_tokens': len(self.eligible),
                'maximum_possible_patch_age': int(self.ages.max())}
