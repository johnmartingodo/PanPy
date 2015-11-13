import numpy as np
from scipy import interpolate
import Mesh

class PanelObject():
	def __init__(self, mesh):
		# Geometry data
		self.mesh = mesh
		self.mesh.calculateFaceCoordinateSystem()

		# Boundary condition type
		self.boundaryType = 'newmann'

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
		self.boundaryType = 'dirichlet'
		self.trailingEdge   = trailingEdge # Index of trailing edges in wing mesh, based on user input
		
		# Default wake settings
		self.wake_nrStrips        = len(self.trailingEdge)
		self.wake_length          = 25
		self.wake_panelLength     = 0.25
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
		self.wake_mesh.calculateFaceCoordinateSystem()

	def addViscousStrips(self, strip_topology, CL, CD):
		self.strip_topology = strip_topology
		self.strip_CL = CL
		self.strip_CD = CD

	def calculateVelocityAndPressure(self, Uinf):
		self.Cp = np.zeros(self.mesh.nrFaces)
		self.u  = np.zeros(self.mesh.nrFaces)
		self.v  = np.zeros(self.mesh.nrFaces)
		self.w  = np.zeros(self.mesh.nrFaces)
		self.U  = np.zeros(self.mesh.nrFaces)

		# Calculate velocity due to sources
		for i in range(self.mesh.nrFaces):
			self.u[i] = 4*np.pi*self.mesh.face_n[i, 0]*self.source_strength[i]
			self.v[i] = 4*np.pi*self.mesh.face_n[i, 1]*self.source_strength[i]
			self.w[i] = 4*np.pi*self.mesh.face_n[i, 2]*self.source_strength[i]

		# Calculate velocity due to doublets
		for i in range(self.mesh.nrFaces):
			nrEdges    = self.mesh.face_nrVerts[i]
			face_startIndex = self.mesh.face_startIndex[i]

			localStrength = self.doublet_strength[i]
			localCenter   = self.mesh.face_center[i]
			gradient      = np.zeros(3)
			weight        = np.zeros(3)

			edgeCounter = 0
			for j in range(nrEdges):
				# Locate neighbour face
				edgeIndex       = self.mesh.face_edges[face_startIndex + j]
				edge_startIndex = self.mesh.edge_startIndex[edgeIndex]
				edge_nrFaces    = self.mesh.edge_nrFaces[edgeIndex]

				if edge_nrFaces == 2:
					if self.mesh.edge_faces[edge_startIndex] == i:
						neighbourFaceIndex = self.mesh.edge_faces[edge_startIndex + 1]
					else:
						neighbourFaceIndex = self.mesh.edge_faces[edge_startIndex]

					neighbourStrength = self.doublet_strength[neighbourFaceIndex]
					neighbourCenter   = self.mesh.face_center[neighbourFaceIndex]

					# Vector from local face center to neighbour
					v = neighbourCenter - localCenter
					dl = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
					v /= dl
					# Difference in strength between local face and neighbour
					dmu = neighbourStrength - localStrength
					dmu_dl = dmu/dl

					gradient += v*dmu_dl
					weight   += np.abs(v)

					edgeCounter += 1

			gradient /= edgeCounter

			self.u[i] += -4*np.pi*gradient[0]
			self.v[i] += -4*np.pi*gradient[1]
			self.w[i] += -4*np.pi*gradient[2]

		# Add free stream velocity
		self.u += Uinf[0]
		self.v += Uinf[1]
		self.w += Uinf[2]

		self.U = np.sqrt(self.u**2 + self.v**2 + self.w**2)

		self.Cp = 1-self.U**2/np.sqrt(Uinf[0]**2 + Uinf[1]**2 + Uinf[2]**2)

	def calculateForces(self):
		self.force = np.zeros(3)

		for i in range(self.mesh.nrFaces):
			self.force[0] += -self.Cp[i]*self.mesh.face_n[i, 0]*self.mesh.face_area[i]
			self.force[1] += -self.Cp[i]*self.mesh.face_n[i, 1]*self.mesh.face_area[i]
			self.force[2] += -self.Cp[i]*self.mesh.face_n[i, 2]*self.mesh.face_area[i]

	def getWakeVerticesInStrip(self, stripNumber):
		indices = np.zeros(self.wake_nrVertsPrStrip, dtype=np.int)

		for i in range(self.wake_nrVertsPrStrip):
			indices[i] = stripNumber*self.wake_nrVertsPrStrip + i

		return indices

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

def generateFlatWingFrom2DProfile(xFoil, yFoil, Span, nrStrips, nrSegments, x0 = 0, y0 = 0, taperRatio = 1, spanDirection = 2):
	# Divide foil into top and bottom
	nrPoints = len(xFoil)

	searchingNose = True
	i = 0
	while searchingNose and i < nrPoints:
		if xFoil[i] < 0.25 and yFoil[i] <= 0:
			iBotStart = i
			searchingNose = False

		i += 1

	x_top = xFoil[0:iBotStart+1]
	x_bot = xFoil[iBotStart:]
	y_top = yFoil[0:iBotStart+1]
	y_bot = yFoil[iBotStart:]

	# Generate top and bottom spline
	topSpl = interpolate.splrep(x_top[::-1], y_top[::-1])
	botSpl = interpolate.splrep(x_bot, y_bot)

	# Generate new x and y values
	x = np.linspace(1, 0, nrSegments+1)
	y = 1*(interpolate.splev(x, topSpl) + interpolate.splev(x, botSpl))
	nrPoints = len(x)

	nrVerts = nrPoints*(nrStrips + 1)
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
		for j in range(nrPoints):
			vert_index = i*nrPoints + j

			verts[vert_index, x_index] = x[j]
			verts[vert_index, y_index] = y[j]
			verts[vert_index, z_index] = z[i]

	# Set right topology
	k = 0
	for i in range(nrStrips):
		for j in range(nrSegments):
			face_verts[k]     = i*nrPoints + j
			face_verts[k + 1] = i*nrPoints + j + 1
			face_verts[k + 2] = (i+1)*nrPoints + j + 1
			face_verts[k + 3] = (i+1)*nrPoints + j
			
			k += 4

	# Initialize mesh
	mesh = Mesh.Mesh(verts, face_verts, face_nrVerts)

	# Initialize PanelObject
	wing = PanelObject(mesh)

	# Find the trailing edge
	tralingVert = np.zeros(nrStrips + 1)
	for i in range(nrStrips + 1):
		tralingVert[i] = i*nrPoints

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

	return wing
