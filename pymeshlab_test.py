import pymeshlab as ml

mesh = ml.MeshSet()
mesh.load_new_mesh('Unsectioned_LowerJaw.stl')
mesh.load_new_mesh('Tooth_{30_94666_20240602_1358}.stl')
mesh.generate_boolean_intersection(first_mesh=0, second_mesh=1)
mesh.save_current_mesh('intersection.stl')
