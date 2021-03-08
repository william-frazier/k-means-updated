# Testing file for Project 1

import unittest
import clustering
from random import Random

class K_meansTest(unittest.TestCase):
    def test_l2_squared_1d(self):
        x = [10]
        y = [15]
        self.assertEqual(
            clustering.calc_l2sq_norm(x,y), 25
        )

    def test_l2_squared_5d(self):
        x = [3, 4, 1, 25, 6]
        y = [6, 4, 3, 2, 10]
        self.assertEqual(
            clustering.calc_l2sq_norm(x,y), 558
        )
        
    def test_l2_squared_neg(self):
        x = [-3, 4, 1, 25, 6]
        y = [6, -4, 3, 2, 10]
        self.assertEqual(
            clustering.calc_l2sq_norm(x,y), 694
        )
        
    def test_importX(self):
        self.assertEqual(
                clustering.import_X("testX.csv")[1][5], 138 
                )
        
    def test_cluster_assignment_one_center(self):
        self.assertEqual(
                clustering.assign_to_centroid([[10,5],[3,-2],[6,18],[-1,-4],[2,2]],[[2,2]]), {0:[[10,5],[3,-2],[6,18],[-1,-4],[2,2]]}
                )
        
    def test_cluster_assignment_several_centers(self):
        self.assertEqual(
                clustering.assign_to_centroid([[1,1],[2,2],[-3,-2],[4,4],[-1,0],[-2,0]],[[2,2],[-3,-2]]), {0:[[1,1],[2,2],[4,4]],1:[[-3,-2],[-1,0],[-2,0]]}
                )
        
    def test_compute_new_centroid_1d(self):
        self.assertEqual(
                clustering.compute_new_centroid([[1],[4.5],[11]]), [5.5]
                )
        
    def test_compute_new_centroid_3d(self):
        self.assertEqual(
            clustering.compute_new_centroid([[1,-1,8],[4.5,1,-6],[11,0,-0.5]]), [5.5,0,0.5]
            )
        
    def test_compute_cost_1d(self):
        self.assertEqual(
                clustering.compute_cost([[1],[3]], {0:[[1],[1.5]],1:[[3],[10],[4]]}),50.25
                )
        
    def test_compute_cost_3d(self):
        self.assertEqual(
                clustering.compute_cost([[1,-1,4],[3,0,-4]], {0:[[1,-1,4],[1.5,-2,2]],1:[[3,0,-4],[10,-1,-5],[4,10,-1]]}), 166.25
                )
        
        

unittest.main()