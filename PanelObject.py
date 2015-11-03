import numpy as np
from scipy import interpolate
import Mesh

class PanelObject():
	def __init__(self, mesh):
		# Geometry data
		self.mesh = mesh

		# Boundary condition type
		self.boundaryType = 'dirichlet'

		# Not a lifting surface by default
		self.liftingSurface = False

		# Data storage arrays for pressure, velocity and panel strength
		self.source_strength  = np.zeros(self.mesh.nrFaces)
		self.doublet_strength = np.zeros(self.mesh.nrFaces)
		self.Cp               = np.zeros(self.mesh.nrFaces)
		self.u                = np.zeros(self.mesh.nrFaces)
		self.v                = np.zeros(self.mesh.nrFaces)
		self.w                = np.zeros(self.mesh.nrFaces)
		self.U                = np.zeros(self.mesh.nrFaces)
		
	def makeLiftinSurface(self, trailingEdge):
		# Makes the panel object a lifting surface, with necessary wake and trailing edge info
		self.liftingSurface = True
		self.trailingEdge   = trailingEdge # Index of trailing edges in wing mesh, based on user input
		
		# Default wake settings
		self.wake_nrStrips        = len(self.trailingEdge)
		self.wake_length          = 20
		self.wake_panelLength     = 1
		self.wake_nrPanelsPrStrip = int(np.ceil(self.wake_length/self.wake_panelLength))
		self.wake_nrVertsPrStrip  = self.wake_nrPanelsPrStrip + 1
		self.wake_trailingVerts   = np.zeros((self.wake_nrStrips, 2), dtype=np.int)


		# List of faces which are connected to the trailing edge. Maximum two
		self.trailingFaces = -np.ones((self.wake_nrStrips, 2), dtype=np.int)

		# Find trailing edge vertices
		self.trailingVerts = np.array([]) # Index of vertices in wing mesh at the trailing edge

		for i in range(self.wake_nrStrips):
			# Look up verts in current edge
			edge = self.mesh.edge_verts[self.trailingEdge[i]]

			# Add missing verts
			if not(edge[0] in self.trailingVerts):
				self.trailingVerts = np.append(self.trailingVerts, edge[0])
			if not(edge[1] in self.trailingVerts):
				self.trailingVerts = np.append(self.trailingVerts, edge[1])

			# Find location of verts in trailingEdge_verts list
			self.wake_trailingVerts[i, 0] = np.where(self.trailingVerts == edge[0])[0][0]
			self.wake_trailingVerts[i, 1] = np.where(self.trailingVerts == edge[1])[0][0]

			edge_index      = self.trailingEdge[i]
			edge_startIndex = self.mesh.edge_startIndex[edge_index]

			self.trailingFaces[i, 0] = self.mesh.edge_faces[edge_startIndex]
			if self.mesh.edge_nrFaces[edge_index] > 1:
				self.trailingFaces[i, 1] = self.mesh.edge_faces[edge_startIndex + 1]

		# Create wake mesh topology and intial data
		self.wake_nrTrailingVerts = len(self.trailingVerts)
		nrVerts = self.wake_nrVertsPrStrip*self.wake_nrTrailingVerts
		nrFaces = self.wake_nrPanelsPrStrip*self.wake_nrStrips

		verts        = np.zeros((nrVerts, 3))
		face_verts   = np.zeros(4*nrFaces)
		face_nrVerts = 4*np.ones(nrFaces)

		k = 0
		for i in range(self.wake_nrStrips):
			# Index of trailingVerts
			index = self.wake_trailingVerts[i]

			for j in range(self.wake_nrPanelsPrStrip):
				face_verts[k]     = index[0]*self.wake_nrVertsPrStrip + j
				face_verts[k + 1] = index[0]*self.wake_nrVertsPrStrip + j + 1
				face_verts[k + 2] = index[1]*self.wake_nrVertsPrStrip + j + 1
				face_verts[k + 3] = index[1]*self.wake_nrVertsPrStrip + j

				k += 4

		# Generate a mesh for the wake
		self.wake_mesh = Mesh.Mesh(verts, face_verts, face_nrVerts, simple=True)

		self.wake_strength = np.zeros(self.wake_mesh.nrFaces)

	def initializeWake(self, direction):
		for i in range(self.wake_nrTrailingVerts):
			v0 = self.mesh.verts[self.trailingVerts[i]]

			for j in range(self.wake_nrVertsPrStrip):
				self.wake_mesh.verts[i*self.wake_nrVertsPrStrip + j] = v0 + j*self.wake_panelLength*direction

		self.wake_mesh.calculateFaceData()

	def addViscousStrips(self, strip_topology, CL, CD):
		self.strip_topology = strip_topology
		self.strip_CL = CL
		self.strip_CD = CD

	def calculateForces(self):
		self.force = np.zeros(3)

		for i in range(self.mesh.nrFaces):
			self.force = -self.Cp[i]*self.mesh.face_n[i]*self.mesh.face_area[i]

def generateWingFrom2DProfile(x, y, Span, nrStrips, x0 = 0, y0 = 0, taperRatio = 1, spanDirection = 2):
	# Close trailing edge gap if it exist
	if y[0] != y[-1]:
		y[0] = 0.5*(y[0] + y[-1])
	if x[0] != x[-1]:
		x[0] != 0.5*(x[0] + x[-1])

	# Calculat enumber of verts and faces
	nrSegments = len(x) - 1

	nrVerts = nrSegments*(nrStrips + 1)
	nrFaces = nrSegments*nrStrips

	# Initialize mesh variables
	verts        = np.zeros((nrVerts, 3))
	face_verts   = np.zeros(4*nrFaces)
	face_nrVerts = 4*np.ones(nrFaces)

	# Set coordinate index
	if spanDirection == 2:
		x_index = 0
		y_index = 1
		z_index = 2
	elif spanDirection == 1:
		x_index = 0
		y_index = 2
		z_index = 1
	elif spanDirection == 0:
		x_index = 2
		y_index = 1
		z_index = 0

	# Calculate span coordinates
	z = np.linspace(0, Span, nrStrips + 1)

	# set right coordinates in mesh verts
	for i in range(nrStrips + 1):
		for j in range(nrSegments):
			vert_index = i*nrSegments + j

			verts[vert_index, x_index] = x[j]
			verts[vert_index, y_index] = y[j]
			verts[vert_index, z_index] = z[i]

	# Set right topology
	k = 0
	for i in range(nrStrips):
		for j in range(nrSegments):
			if j == nrSegments - 1:
				face_verts[k]     = i*nrSegments + j
				face_verts[k + 1] = i*nrSegments
				face_verts[k + 2] = (i+1)*nrSegments
				face_verts[k + 3] = (i+1)*nrSegments + j
			else:
				face_verts[k]     = i*nrSegments + j
				face_verts[k + 1] = i*nrSegments + j + 1
				face_verts[k + 2] = (i+1)*nrSegments + j + 1
				face_verts[k + 3] = (i+1)*nrSegments + j
			k += 4

	# Initialize mesh
	mesh = Mesh.Mesh(verts, face_verts, face_nrVerts)

	# Initialize PanelObject
	wing = PanelObject(mesh)

	# Find the trailing edge
	tralingVert = np.zeros(nrStrips + 1)
	for i in range(nrStrips + 1):
		tralingVert[i] = i*nrSegments

	trailingEdge = np.zeros(nrStrips)
	i_current = 0
	for i in range(wing.mesh.nrEdges):
		edge_vert = wing.mesh.edge_verts[i]

		if (edge_vert[0] in tralingVert) and (edge_vert[1] in tralingVert):
			trailingEdge[i_current] = i
			i_current += 1

	wing.makeLiftinSurface(trailingEdge)
	wing.initializeWake(np.array([1, 0, 0]))

	return wing




