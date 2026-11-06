import unittest
import numpy as np
from my_harness import Controller


class StallRefreshTests(unittest.TestCase):
    def test_stationary_measured_motion_refreshes_three_calls_then_cools_down(self):
        c=Controller('stall_refresh');image=np.zeros((224,224,3),dtype=np.uint8)
        pos=np.zeros(3);fingers=np.zeros(2);action=np.array([.03,0,0,0,0,0,0])
        for _ in range(8):
            self.assertFalse(c.before(image,pos,fingers))
            c.after(action,image,False)
        for _ in range(3):
            self.assertTrue(c.before(image,pos,fingers))
            c.after(action,image,False)
        self.assertFalse(c.before(image,pos,fingers))

    def test_observed_progress_does_not_trigger_stall_refresh(self):
        c=Controller('stall_refresh');image=np.zeros((224,224,3),dtype=np.uint8)
        action=np.array([.03,0,0,0,0,0,0])
        for i in range(12):
            self.assertFalse(c.before(image,np.array([i*.01,0,0]),np.zeros(2)))
            c.after(action,image,False)


if __name__=='__main__':unittest.main()
