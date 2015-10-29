#!python
#cython: language_level=3, boundscheck=False, wraparound=True

import numpy as np
cimport numpy as np

cimport Mesh

cimport cython    

cdef class Mesh:
	def __cinit__(self, verts, face_verts, face_nrVerts, simple=False):
		cdef int i, j
		# Vertices data
		self.nrVerts = len(verts)
		self.verts = verts

		# Face data
		self.nrFaces         = len(face_nrVerts)
		self.face_verts      = face_verts.astype(np.int)
		self.face_nrVerts    = face_nrVerts.astype(np.int)

		self.face_startIndex = np.zeros(self.nrFaces, dtype=int)
		for i in range(1, self.nrFaces):
			self.face_startIndex[i] = self.face_startIndex[i-1] + self.face_nrVerts[i-1]

		if not simple:
			self.updateMeshData()

		# Custom face data
		self.face_dataNames = []

	# --------- Public accessible variables -------------------------------
	property nrVerts:
		def __get__(self):
			return self.nrVerts
	property verts:
		def __get__(self):
			return np.asarray(self.verts)
		def __set__(self, np.ndarray[dtype=np.float_t, ndim=2] vert_inn):
			self.verts = vert_inn
	property vert_n:
		def __get__(self):
			return np.asarray(self.vert_n)

	property nrFaces:
		def __get__(self):
			return self.nrFaces	
	property face_verts:
		def __get__(self):
			return np.asarray(self.face_verts)
	property face_nrVerts:
		def __get__(self):
			return np.asarray(self.face_nrVerts)
	property face_startIndex:
		def __get__(self):
			return np.asarray(self.face_startIndex)
	property face_edges:
		def __get__(self):
			return np.asarray(self.face_edges)
	property face_l:
		def __get__(self):
			return np.asarray(self.face_l)
	property face_m:
		def __get__(self):
			return np.asarray(self.face_m)
	property face_n:
		def __get__(self):
			return np.asarray(self.face_n)
	property face_area:
		def __get__(self):
			return np.asarray(self.face_area)
	property face_center:
		def __get__(self):
			return np.asarray(self.face_center)
	property face_dataNames:
		def __get__(self):
			return self.face_dataNames
	property face_data:
		def __get__(self):
			return np.asarray(self.face_data)
	
	property nrEdges:
		def __get__(self):
			return self.nrEdges
	property edge_verts:
		def __get__(self):
			return np.asarray(self.edge_verts)
	property edge_faces:
		def __get__(self):
			return np.asarray(self.edge_faces)
	property edge_nrFaces:
		def __get__(self):
			return np.asarray(self.edge_nrFaces)
	property edge_startIndex:
		def __get__(self):
			return np.asarray(self.edge_startIndex)

	# ---------- Methods --------------------------------------------------
	def calculateFaceData(self):
		cdef int i, j, startIndex, stopIndex, nrVerts, i1, i2
		cdef np.ndarray[dtype=np.int_t, ndim=1]    indices
		cdef np.ndarray[dtype=np.double_t, ndim=1] cross
		cdef np.ndarray[dtype=np.double_t, ndim=1] n
		cdef np.ndarray[dtype=np.double_t, ndim=1] v1
		cdef np.ndarray[dtype=np.double_t, ndim=1] v2

		self.face_n      = np.zeros((self.nrFaces, 3), dtype=np.double)
		self.face_area   = np.zeros( self.nrFaces,     dtype=np.double)
		self.face_center = np.zeros((self.nrFaces, 3), dtype=np.double)

		for i in range(self.nrFaces):
			cross = np.zeros(3, dtype=np.double)
			v1    = np.zeros(3, dtype=np.double)
			v2    = np.zeros(3, dtype=np.double)

			startIndex = self.face_startIndex[i]
			nrVerts    = self.face_nrVerts[i]
			stopIndex  = startIndex + nrVerts
			
			indices = np.asarray(self.face_verts[startIndex:stopIndex])

			# Calculate normal
			if nrVerts == 3:
				for j in range(3):
					v1[j] = self.verts[indices[1], j] - self.verts[indices[0], j]
					v2[j] = self.verts[indices[2], j] - self.verts[indices[0], j]
			else:
				for j in range(3):
					v1[j] = self.verts[indices[2], j] - self.verts[indices[0], j]
					v2[j] = self.verts[indices[3], j] - self.verts[indices[1], j]

			# Calculate cross product
			n = np.cross(v1, v2)

			# Normalize normal
			l = np.sqrt(np.sum(n**2))
			n /= l

			self.face_n[i, 0] = n[0]
			self.face_n[i, 1] = n[1]
			self.face_n[i, 2] = n[2]

			# Calculate area
			for j in range(self.face_nrVerts[i]):
				i1 = indices[j]

				v1 = np.asarray(self.verts[i1])

				self.face_center[i, 0] += v1[0]
				self.face_center[i, 1] += v1[1]
				self.face_center[i, 2] += v1[2]

				if j == nrVerts-1:
					i2 = indices[0]
				else:
					i2 = indices[j+1]

				v2 = np.asarray(self.verts[i2])

				cross += np.cross(v1, v2)

			self.face_area[i] = 0.5*np.dot(n, cross)

			self.face_center[i, 0] /= nrVerts
			self.face_center[i, 1] /= nrVerts
			self.face_center[i, 2] /= nrVerts

	def calculateFaceCoordinateSystem(self):
		cdef int i, j, i_start, i1, i2
		cdef double l_length, m_length

		cdef np.ndarray[dtype=np.double_t, ndim=1] l
		cdef np.ndarray[dtype=np.double_t, ndim=1] m
		cdef np.ndarray[dtype=np.double_t, ndim=1] n
		cdef np.ndarray[dtype=np.double_t, ndim=1] v0
		cdef np.ndarray[dtype=np.double_t, ndim=1] v1
		cdef np.ndarray[dtype=np.double_t, ndim=1] v2
		cdef np.ndarray[dtype=np.double_t, ndim=1] v3
		cdef np.ndarray[dtype=np.double_t, ndim=1] v4

		self.face_l = np.zeros((self.nrFaces, 3), dtype=np.double)
		self.face_m = np.zeros((self.nrFaces, 3), dtype=np.double)

		for i in range(self.nrFaces):
			# Look up data
			i_start = self.face_startIndex[i]
			i1      = self.face_verts[i_start]
			i2      = self.face_verts[i_start + 1]
			v0      = np.asarray(self.face_center[i])
			v1      = np.asarray(self.verts[i1])
			v2      = np.asarray(self.verts[i2])

			# Calculate local l, m, based on data
			n = np.asarray(self.face_n[i])

			if self.face_nrVerts[i] == 4:
				v1 = np.asarray(self.verts[self.face_verts[i_start]])
				v2 = np.asarray(self.verts[self.face_verts[i_start + 1]])
				v3 = np.asarray(self.verts[self.face_verts[i_start + 2]])
				v4 = np.asarray(self.verts[self.face_verts[i_start + 3]])

				l = 0.5*(v2 + v3) - 0.5*(v1 + v4)
			else:
				l = 0.5*(v2 + v1) - v0

			l_length = np.sqrt(np.sum(l**2))
			l /= l_length

			m = np.cross(n, l)
			m_length = np.sqrt(np.sum(m**2))
			m /= m_length

			# Add calculated m and l to array
			for j in range(3):
				self.face_m[i, j] = m[j]
				self.face_l[i, j] = l[j]

	def calculateEdgeConnectivity(self):
		cdef int startIndex, edge_startIndex
		cdef int v1
		cdef int v2
		cdef int edgeExist
		cdef int i, j, k, offset, edgeIndex

		self.nrEdges = 0

		self.edge_verts      = np.array([[]], dtype=np.int)
		self.edge_faces      = np.array([],   dtype=np.int)
		self.edge_nrFaces    = np.array([],   dtype=np.int)
		self.edge_startIndex = np.array([],   dtype=np.int)

		self.face_edges = np.zeros(len(self.face_verts), dtype=np.int)

		# Go through all the faces
		for i in range(self.nrFaces):
			startIndex = self.face_startIndex[i]

			# Go through all the vertices in the face
			for j in range(self.face_nrVerts[i]):

				# Find vertices that make up an edge
				v1 = self.face_verts[startIndex+j]

				if j == self.face_nrVerts[i]-1:
					v2 = self.face_verts[startIndex]
				else:
					v2 = self.face_verts[startIndex+j+1]

				# Check if edge data allready exist in stored values
				edgeExist = 0

				checkEdge = np.zeros(self.nrEdges, dtype=np.int)
				for k in range(self.nrEdges):

					if (self.edge_verts[k, 0] == v1 or self.edge_verts[k, 0] == v2) and (self.edge_verts[k, 1] == v1 or self.edge_verts[k, 1] == v2):
						edgeExist = 1
						edgeIndex = k
						break

				# If edge does not exist, add it to edge list
				if edgeExist == 0:
					if self.nrEdges == 0:
						self.edge_verts = np.array([[v1, v2]], dtype=np.int)
					else:
						self.edge_verts = np.vstack([self.edge_verts, np.array([v1, v2])])

					edgeIndex = self.nrEdges
		
					self.nrEdges += 1

					self.edge_nrFaces = np.append(self.edge_nrFaces, 0)

				self.face_edges[startIndex + j] = edgeIndex

				# Add face to edge data
				self.edge_nrFaces[edgeIndex] += 1

		# Find start index for edge_face data
		self.edge_startIndex = np.zeros(self.nrEdges, dtype=np.int)
		for i in range(1, self.nrEdges):
			self.edge_startIndex[i] = self.edge_startIndex[i-1] + self.edge_nrFaces[i-1]

		# Find edge_faces
		self.edge_faces = np.zeros(np.sum(self.edge_nrFaces), dtype=np.int)
		offsetArray = np.zeros(self.nrEdges, dtype=np.int)

		# Go through each face
		for i in range(self.nrFaces):
			startIndex = self.face_startIndex[i]

			# GO through each edge (vertex) in that face
			for j in range(self.face_nrVerts[i]):
				# Find index of current edge
				edgeIndex = self.face_edges[startIndex+j]
				edge_startIndex = self.edge_startIndex[edgeIndex]

				# Add face index to right place in edge_faces array
				offset = offsetArray[edgeIndex]
				self.edge_faces[edge_startIndex + offset] = i
				offsetArray[edgeIndex] += 1

	def updateMeshData(self):
		self.calculateFaceData()
		self.calculateEdgeConnectivity()

	def triangulate(self):
		cdef int i, j
		cdef np.ndarray[dtype=np.int_t, ndim=1] face_vertsNew = np.array([], dtype=np.int)

		cdef np.ndarray[dtype=np.int_t, ndim=1] indices

		cdef np.ndarray[dtype=np.double_t, ndim=1] dig1 = np.zeros(3, dtype=np.double)
		cdef np.ndarray[dtype=np.double_t, ndim=1] dig2 = np.zeros(3, dtype=np.double)

		cdef np.ndarray[dtype=np.int_t, ndim=1] face1
		cdef np.ndarray[dtype=np.int_t, ndim=1] face2

		cdef double l1
		cdef double l2

		cdef int startIndex
		cdef int center_index
		cdef int index1
		cdef int index2

		for i in range(self.nrFaces):
			startIndex = self.face_startIndex[i]

			indices = np.zeros(self.face_nrVerts[i], dtype=np.int)
			for j in range(self.face_nrVerts[i]):
				indices[j] = self.face_verts[startIndex + j]

			if self.face_nrVerts[i] > 3:
				
				if self.face_nrVerts[i] == 4:
					# Quad, split shortest diagonal
					for j in range(3):
						dig1[j] = self.verts[indices[2], j] - self.verts[indices[0], j]
						dig2[j] = self.verts[indices[3], j] - self.verts[indices[1], j]

					l1 = np.sqrt(np.sum(dig1**2))
					l2 = np.sqrt(np.sum(dig2**2))

					if l1 < l2:
						face1 = np.array([indices[0], indices[1], indices[2]], dtype=np.int)
						face2 = np.array([indices[0], indices[2], indices[3]], dtype=np.int)
					else:
						face1 = np.array([indices[0], indices[1], indices[3]], dtype=np.int)
						face2 = np.array([indices[1], indices[2], indices[3]], dtype=np.int)

					face_vertsNew = np.append(face_vertsNew, face1)
					face_vertsNew = np.append(face_vertsNew, face2)

				else:
					# Triangle fan around center
					self.verts = np.append(self.verts, [self.face_center[i]], axis=0)
					self.nrVerts += 1
					center_index = len(self.vertices)-1

					for j in range(self.face_nrVerts[i]):
						index1 = indices[j]

						if j == self.face_nrVerts[i]-1:
							index2 = indices[0]
						else:
							index2 = indices[j+1]

						face1 = np.array([center_index, index1, index2])
						face_vertsNew = np.append(face_vertsNew, face1)
				
			else:
				face_vertsNew = np.append(face_vertsNew, indices)

		self.face_verts = face_vertsNew
		self.nrFaces = int(len(self.face_verts)/3)

		self.face_nrVerts    = 3*np.ones(self.nrFaces, dtype=np.int)
		self.face_startIndex =  np.zeros(self.nrFaces, dtype=np.int)

		for i in range(1, self.nrFaces):
			self.face_startIndex[i] = self.face_startIndex[i-1] + 3

		self.updateMeshData()

	def addFaceData(self, face_dataName, face_data):
		existingFaceData = False

		if len(self.face_dataNames) > 0:
			existingFaceData = True

		self.face_dataNames.append(face_dataName)

		if existingFaceData:
			self.face_data = np.vstack((self.face_data, face_data))
		else:
			self.face_data = np.zeros((1, self.nrFaces), dtype=np.double)

			for i in range(self.nrFaces):
				self.face_data[0, i] = face_data[i]

	def scale(self, scaleVector, scaleCenter=np.zeros(3)):
		cdef int i
		for i in range(self.nrVerts):
			self.verts[i, 0] = (self.verts[i, 0] - scaleCenter[0])*scaleVector[0] + scaleVector[0]
			self.verts[i, 1] = (self.verts[i, 1] - scaleCenter[1])*scaleVector[1] + scaleVector[1]
			self.verts[i, 2] = (self.verts[i, 2] - scaleCenter[2])*scaleVector[2] + scaleVector[2]

	def translate(self, vector):
		cdef int i
		for i in range(self.nrVerts):
			self.verts[i, 0] += vector[0]
			self.verts[i, 1] += vector[1]
			self.verts[i, 2] += vector[2]

	def rotate(self, angleVector, rotationCenter=np.zeros(3)):
		cdef int i
		cdef double angle
		cdef np.ndarray[dtype=np.double_t, ndim=2] Rx
		cdef np.ndarray[dtype=np.double_t, ndim=2] Ry
		cdef np.ndarray[dtype=np.double_t, ndim=2] Rz
		cdef np.ndarray[dtype=np.double_t, ndim=1] vert

		# Rotation matrix x axis
		angle = angleVector[0]
		Rx = np.array([[1, 0,                          0],
					   [0, np.cos(angle), -np.sin(angle)],
					   [0, np.sin(angle),  np.cos(angle)]])

		# Rotation matrix y axis
		angle = angleVector[1]
		Ry = np.array([[ np.cos(angle), 0, np.sin(angle)],
					   [0,              1,              0],
					   [-np.sin(angle), 0, np.cos(angle)]])

		# Rotation matrix z axis
		angle = angleVector[2]
		Rz = np.array([[np.cos(angle), -np.sin(angle), 0],
					   [np.sin(angle),  np.cos(angle), 0],
					   [0,              0,             1]])

		for i in range(self.nrVerts):
			vert = np.asarray(self.verts[i])

			vert -= rotationCenter
			vert = np.dot(Rx, vert)
			vert = np.dot(Ry, vert)
			vert = np.dot(Rz, vert)
			vert += rotationCenter

			self.verts[i, 0] = vert[0]
			self.verts[i, 1] = vert[1]
			self.verts[i, 2] = vert[2]

	# ----- Export -------------------------
	def exportObj(self, filePath, exportNormals=False):
		cdef int i, j, startIndex
		# create file
		f = open(filePath, 'w')

		# write header
		f.write('# Exported from pyMesh by Jarle Kramer\n')
		f.write("o object\n")

		# Write vertices
		for i in range(self.nrVerts):
			f.write('v {:.6f} {:.6f} {:.6f}\n'.format(self.verts[i][0], self.verts[i][1], self.verts[i][2]))

		# Write face normals
		if exportNormals:
			for i in range(self.nrFaces):
				f.write('vn {:.6f} {:.6f} {:.6f}\n'.format(self.face_n[i][0], self.face_n[i][1], self.face_n[i][2]))

		# Write faces
		for i in range(self.nrFaces):
			startIndex = self.face_startIndex[i]

			f.write('f ')
			for j in range(self.face_nrVerts[i]):
				if exportNormals:
					f.write('{:.0f}//{:.0f} '.format(self.face_verts[startIndex+j] + 1, i+1))
				else:
					f.write('{:.0f} '.format(self.face_verts[startIndex+j] + 1))

			f.write('\n')

		f.close()

	def exportVTK(self, filePath):
		cdef int i, j

		f = open(filePath, 'w')

		# Write header
		f.write('<?xml version="1.0"?>\n')
		f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
		f.write('\t<PolyData>\n')
		f.write('\t\t<Piece NumberOfPoints="{:.0f}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{:.0f}">\n'.format(self.nrVerts, self.nrFaces))
		
		# Write vertices
		f.write('\t\t\t<Points>\n')
		f.write('\t\t\t\t<DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
		f.write('\t\t\t\t\t')
		for i in range(self.nrVerts):
			for j in range(3):
				f.write('{:.6f} '.format(self.verts[i][j]))

		f.write('\n')
		f.write('\t\t\t\t</DataArray>\n')
		f.write('\t\t\t</Points>\n')

		# Write faces
		f.write('\t\t\t<Polys>\n')
		f.write('\t\t\t\t<DataArray type="Int32" Name="connectivity" format="ascii">\n')
		f.write('\t\t\t\t\t')
		for i in range(len(self.face_verts)):
			f.write('{:.0f} '.format(self.face_verts[i]))
		f.write('\n')
		f.write('\t\t\t\t</DataArray>\n')
		f.write('\t\t\t\t<DataArray type="Int32" Name="offsets" format="ascii">\n')
		f.write('\t\t\t\t\t')
		for i in range(self.nrFaces):
			f.write('{:.0f} '.format(self.face_startIndex[i] + self.face_nrVerts[i]))
		f.write('\n')
		f.write('\t\t\t\t</DataArray>\n')
		f.write('\t\t\t</Polys>\n')

		# Write face data
		if len(self.face_dataNames) > 0:
			f.write('\t\t\t<CellData Scalars="')
			for i in range(len(self.face_dataNames)):
				f.write(self.face_dataNames[i])
				if i != len(self.face_dataNames)-1:
					f.write(' ')

			f.write('">\n')

			for i in range(len(self.face_dataNames)):
				f.write('\t\t\t\t<DataArray type="Float32" Name="{}" format="ascii">\n'.format(self.face_dataNames[i]))
				f.write('\t\t\t\t\t')
				for j in range(self.nrFaces):
					f.write('{:.6f} '.format(self.face_data[i, j]))
				f.write('\n')
				f.write('\t\t\t\t</DataArray>\n')
				
			f.write('\t\t\t</CellData>\n')

		# Write footer
		f.write('\t\t</Piece>\n')
		f.write('\t</PolyData>\n')
		f.write('</VTKFile>\n')

		f.close()

# ---------------- Functions to create mesh from file ---------------------------------
def importObj(filePath, simple=False):
	cdef int  i, j, nrVerts, lineLength, firstVert
	cdef str  line
	cdef list lineList
	cdef np.ndarray[dtype=np.float_t, ndim=2] verts
	cdef np.ndarray[dtype=np.float_t, ndim=1] vertData
	cdef np.ndarray[dtype=np.int_t, ndim=1]    faceData
	cdef np.ndarray[dtype=np.int_t, ndim=1]    face_verts   = np.array([], dtype=np.int)
	cdef np.ndarray[dtype=np.int_t, ndim=1]    face_nrVerts = np.array([], dtype=np.int)

	# Open file
	f = open(filePath, 'r')

	firstVert = 1

	while True:
		line = f.readline()
		# Break at the end of line
		if not line:
			break

		if line.strip():
			lineList = line.strip().split()

			# Check for vertex line
			if lineList[0] == 'v':
				vertData = np.array([float(lineList[1]), float(lineList[2]), float(lineList[3])])
					
				if firstVert:
					verts = np.array([vertData], dtype=np.float)
					firstVert = 0
				else:
					verts = np.append(verts, np.array([vertData]), axis=0)

			# Check for face line
			elif lineList[0] == 'f':
				lineLength = len(lineList)
				nrVerts = lineLength-1
				faceData = np.zeros(nrVerts, dtype=int)

				for j in range(nrVerts):
					faceData[j] = int(lineList[j+1].split('//')[0])-1

				face_verts   = np.append(face_verts, faceData)
				face_nrVerts = np.append(face_nrVerts, nrVerts)

	f.close()

	mesh = Mesh(verts, face_verts, face_nrVerts, simple=simple)

	return mesh
