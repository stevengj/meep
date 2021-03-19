import meep as mp
import copy
import unittest
import numpy as np

chunk_ids = []
chunk_areas = []

def traverse_tree(bp=None,min_corner=None,max_corner=None):
    if ((min_corner.x > max_corner.x) or (min_corner.y > max_corner.y)):
        raise RuntimeError("min_corner/max_corner have been incorrectly defined.")

    ## reached a leaf
    if (bp.left is None and bp.right is None):
        chunk_ids.append(bp.id)
        chunk_area = (max_corner.x-min_corner.x)*(max_corner.y-min_corner.y)
        chunk_areas.append(chunk_area)

    ## traverse the left branch                                      
    if (bp.left is not None):
        new_max_corner = copy.deepcopy(max_corner)
        if bp.split_dir == mp.X:
            new_max_corner.x = bp.split_pos
        else:
            new_max_corner.y = bp.split_pos
        traverse_tree(bp.left,min_corner,new_max_corner)

    ## traverse the right branch                                      
    if (bp.right is not None):
        new_min_corner = copy.deepcopy(min_corner)
        if bp.split_dir == mp.X:
            new_min_corner.x = bp.split_pos
        else:
            new_min_corner.y = bp.split_pos
        traverse_tree(bp.right,new_min_corner,max_corner)


class TestChunkLayoutBinaryPartition(unittest.TestCase):

    def test_chunk_layout_binary_partition(self):
        chunk_layout = mp.BinaryPartition(data=[ (mp.X,-2.0), 0, [ (mp.Y,1.5), [ (mp.X,3.0), 1, [ (mp.Y,-0.5), 4,3 ] ], 2 ] ])

        cell_size = mp.Vector3(10.0,5.0,0)

        sim = mp.Simulation(cell_size=cell_size,
                            resolution=10,
                            num_chunks=5,
                            chunk_layout=chunk_layout)

        sim.init_sim()
        owners = sim.structure.get_chunk_owners()
        areas = [ v.surroundings().full_volume() for v in sim.structure.get_chunk_volumes() ]

        traverse_tree(chunk_layout,-0.5*cell_size,0.5*cell_size)

        np.testing.assert_array_equal(owners,np.asarray(chunk_ids))
        self.assertListEqual(areas,chunk_areas)
        
if __name__ == '__main__':
    unittest.main()
