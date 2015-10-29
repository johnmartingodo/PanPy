import numpy as np
cimport numpy as np

cdef class Mesh:
	cdef:
		# Vertices data
		int nrVerts
		double[:, :] verts  # Coordinates of vertices
		double[:, :] vert_n # Normal coordinates for each vertex

		# Face data
		int          nrFaces
		long[:]      face_verts      # Index of face vertices as 1D array
		long[:]      face_nrVerts    # Number of vertices/edges in each face
		long[:]      face_startIndex # Start index for each face, when accessing the face_vert array
		long[:]      face_edges      # Index of face edges as 1D array
		double[:]    face_area       # Area of face
		double[:, :] face_l          # local x-axis
		double[:, :] face_m          # local y-axis
		double[:, :] face_n          # Normal coordinates for each face and local z-axis
		double[:, :] face_center     # Center of face
		list         face_dataNames  # List of names of custom data
		double[:, :] face_data       # 2D array used to store custom face data

		# Edge data
		int nrEdges
		long[:, :] edge_verts      # Index of edge vertices
		long[:]    edge_faces      # Index of faces connected to edge
		long[:]    edge_nrFaces    # Number of connected faces for each edge
		long[:]    edge_startIndex # Start index for each edge, when accessing the edge_faces array