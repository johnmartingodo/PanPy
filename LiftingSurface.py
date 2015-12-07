import numpy as np
from scipy import interpolate
import Mesh
from PanelObject import *

class LiftingSurface(PanelObject):
	''' Class that inherites from the more general PanelObject class, but with the added structure needed in order to analyze a lifting surface. 
	More restricted in terms of geometry, but should (eventually) cover all normal types of wing geometry. 
	Basically, it must be a "strip" based geometry, from 2D foil profile data, that can follow an arbitrary "span line" 
	Varying foil geometry and twist along the span will eventually be possible.'''

	def __init__(self, x, y, span, nrStrips, spanDirection = 2):
		self.nrStrips      = nrStrips
		self.nrSegments    = len(x) - 1
		self.spanDirection = spanDirection
		self.foil_x = x
		self.foil_y = y
		self.span   = span

		# Default wake settings
		self.wake_length           = 25
		self.wake_panelLength      = 0.25
		self.wake_nrPanelsPrStrip  = int(np.ceil(self.wake_length/self.wake_panelLength))
		self.wake_nrVertsPrStrip   = self.wake_nrPanelsPrStrip + 1
		self.wake_initialDirection = np.array([1, 0, 0]) 

		# Generate wing geoemtry
		self.generateWingGeometry()

		self.liftingSurface = True

		# Structure strip faces so that they easily can be used later
		self.stripFaces = np.zeros((self.nrStrips, self.nrSegments), dtype=np.int)

		for i in range(self.nrStrips):
			for j in range(self.nrSegments):
				self.stripFaces[i, j] = i*self.nrSegments + j
		
		# Structure verts into strips
		self.stripVerts = np.zeros((self.nrStrips+1, self.nrSegments), dtype=np.int)

		for i in range(self.nrStrips+1):
			for j in range(self.nrSegments):
				self.stripVerts[i, j] = i*self.nrSegments + j 


		# Set strip face strength based on distance from lower trailing edge
		self.stripFaceStrength = np.zeros((self.nrStrips, self.nrSegments))
		for i in range(self.nrStrips):
			# Construct spline from strip face centers
			t = np.linspace(0, 1, self.nrSegments)

			x = self.mesh.face_center[self.stripFaces[i], 0]
			y = self.mesh.face_center[self.stripFaces[i], 1]
			z = self.mesh.face_center[self.stripFaces[i], 2]

			# Add trailing edge data to x, y, z
			#x_t = 0.5*(self.mesh.verts[self.stripVerts[i, 0], 0] + self.mesh.verts[self.stripVerts[i+1, 0], 0])

			#y_t = 0.5*(self.mesh.verts[self.stripVerts[i, 0], 1] + self.mesh.verts[self.stripVerts[i+1, 0], 1])
			#z_t = 0.5*(self.mesh.verts[self.stripVerts[i, 0], 2] + self.mesh.verts[self.stripVerts[i+1, 0], 2])

			#x = np.insert(x, 0, x_t)
			#x = np.append(x, x_t)
			#y = np.insert(y, 0, y_t)
			#y = np.append(y, y_t)
			#z = np.insert(z, 0, z_t)
			#z = np.append(z, z_t)

			xSpl = interpolate.splrep(t, x)
			ySpl = interpolate.splrep(t, y)
			zSpl = interpolate.splrep(t, z)

			# Calculate total length of spline
			n = 50
			t_int = np.linspace(0, 1, n)

			dx = interpolate.splev(t_int, xSpl, der=1)
			dy = interpolate.splev(t_int, ySpl, der=1)
			dz = interpolate.splev(t_int, zSpl, der=1)

			integrand = np.sqrt(dx**2 + dy**2 + dz**2)

			L = np.trapz(integrand, x=t_int)

			for j in range(self.nrSegments):
				# Calculate length from upper trailing edge to current face_center
				t_int = np.linspace(0, t[j], n)

				dx = interpolate.splev(t_int, xSpl, der=1)
				dy = interpolate.splev(t_int, ySpl, der=1)
				dz = interpolate.splev(t_int, zSpl, der=1)

				integrand = np.sqrt(dx**2 + dy**2 + dz**2)

				l = np.trapz(integrand, x=t_int)

				self.stripFaceStrength[i, j] = (1 - l/L)

		# Locate trailing edge vertices on the wing
		self.trailingVerts = np.zeros(self.nrStrips + 1)
		for i in range(self.nrStrips + 1):
			self.trailingVerts[i] = i*self.nrSegments

		# Generate wake
		self.generateWakeGeometry()
		self.wake_strength = np.zeros(self.wake_mesh.nrFaces)
		self.wake_stripFaces = np.zeros((self.nrStrips, self.wake_nrPanelsPrStrip), dtype=np.int)
		# Structure strip faces of the wake so that they easily can be used later
		for i in range(self.nrStrips):
			for j in range(self.wake_nrPanelsPrStrip):
				self.wake_stripFaces[i, j] = i*self.wake_nrPanelsPrStrip + j

		# Create kutta condition control points and direction vectors
		self.generateKuttaGeometry()

	def generateWingGeometry(self):
		# Close trailing edge gap if it exist
		if self.foil_y[0] != self.foil_y[-1]:
			self.foil_y[0] = 0.5*(self.foil_y[0] + self.foil_y[-1])
		if self.foil_x[0] != self.foil_x[-1]:
			self.foil_x[0] != 0.5*(self.foil_x[0] + self.foil_x[-1])

		# Calculat enumber of verts and faces
		nrVerts = self.nrSegments*(self.nrStrips + 1)
		nrFaces = self.nrSegments*self.nrStrips

		# Initialize mesh variables
		verts        = np.zeros((nrVerts, 3))
		face_verts   = np.zeros(4*nrFaces)
		face_nrVerts = 4*np.ones(nrFaces)

		# Set coordinate index
		self.span_vector = np.zeros(3)
		self.span_vector[self.spanDirection] = 1
		if self.spanDirection == 2:
			x_index = 0
			y_index = 1
			z_index = 2
		elif self.spanDirection == 1:
			x_index = 0
			y_index = 2
			z_index = 1
		elif self.spanDirection == 0:
			x_index = 2
			y_index = 1
			z_index = 0

		# Calculate span coordinates
		z = np.linspace(0, self.span, self.nrStrips + 1)

		# set right coordinates in mesh verts
		for i in range(self.nrStrips + 1):
			for j in range(self.nrSegments):
				vert_index = i*self.nrSegments + j

				verts[vert_index, x_index] = self.foil_x[j]
				verts[vert_index, y_index] = self.foil_y[j]
				verts[vert_index, z_index] = z[i]

		# Set right topology
		k = 0
		for i in range(self.nrStrips):
			for j in range(self.nrSegments):
				if j == self.nrSegments - 1:
					face_verts[k]     = i*self.nrSegments + j
					face_verts[k + 1] = i*self.nrSegments
					face_verts[k + 2] = (i+1)*self.nrSegments
					face_verts[k + 3] = (i+1)*self.nrSegments + j
				else:
					face_verts[k]     = i*self.nrSegments + j
					face_verts[k + 1] = i*self.nrSegments + j + 1
					face_verts[k + 2] = (i+1)*self.nrSegments + j + 1
					face_verts[k + 3] = (i+1)*self.nrSegments + j
				k += 4

		# Initialize mesh
		mesh = Mesh.Mesh(verts, face_verts, face_nrVerts)

		# Initialize PanelObject
		PanelObject.__init__(self, mesh)

	def generateWakeGeometry(self):
		# Create wake mesh topology and intial data
		nrVerts = self.wake_nrVertsPrStrip*(self.nrStrips + 1)
		nrFaces = self.wake_nrPanelsPrStrip*self.nrStrips

		verts        = np.zeros((nrVerts, 3))
		face_verts   = np.zeros(4*nrFaces)
		face_nrVerts = 4*np.ones(nrFaces)

		# Set vertices coordiantes
		for i in range(self.nrStrips + 1):
			v0 = self.mesh.verts[self.trailingVerts[i]]

			for j in range(self.wake_nrVertsPrStrip):
				verts[i*self.wake_nrVertsPrStrip + j] = v0 + j*self.wake_panelLength*self.wake_initialDirection

		k = 0
		for i in range(self.nrStrips):
			for j in range(self.wake_nrPanelsPrStrip):
				face_verts[k]     = i*self.wake_nrVertsPrStrip + j
				face_verts[k + 1] = (i+1)*self.wake_nrVertsPrStrip + j
				face_verts[k + 2] = (i+1)*self.wake_nrVertsPrStrip + j + 1
				face_verts[k + 3] = i*self.wake_nrVertsPrStrip + j + 1

				k += 4

		# Generate a mesh for the wake
		self.wake_mesh = Mesh.Mesh(verts, face_verts, face_nrVerts, simple=True)
		self.wake_mesh.calculateFaceData()
		self.wake_mesh.calculateFaceCoordinateSystem()

	def generateKuttaGeometry(self):
		self.kutta_ctrlPoints = np.zeros((self.nrStrips, 3))
		self.kutta_direction  = np.zeros((self.nrStrips, 3))
		self.kutta_normal     = np.zeros((self.nrStrips, 3))
		self.kutta_distance   = 0.00001

		for i in range(self.nrStrips):
			v1 = np.zeros(3)
			v2 = np.zeros(3)

			# Locate faces at trailing edge
			i1 = self.stripFaces[i, 0]
			i2 = self.stripFaces[i, -1]

			# Mid point at trailing edge
			p1 = self.mesh.verts[self.trailingVerts[i]]
			p2 = self.mesh.verts[self.trailingVerts[i+1]]

			pe = 0.5*(p1 + p2)

			# Create first vector
			v1 = pe - self.mesh.face_center[i1]

			# Create second vector
			v2 = pe - self.mesh.face_center[i2]

			# Normalize vectors
			v1 /= np.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
			v2 /= np.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

			self.kutta_direction[i] = v1 + v2
			self.kutta_direction[i] /= np.sqrt(self.kutta_direction[i, 0]**2 + self.kutta_direction[i, 1]**2 + self.kutta_direction[i, 2]**2)

			self.kutta_ctrlPoints[i] = pe + self.kutta_direction[i]*self.kutta_distance

			self.kutta_normal[i] = np.cross(self.span_vector, self.kutta_direction[i])
			self.kutta_normal[i] /= np.sqrt(self.kutta_normal[i, 0]**2 + self.kutta_normal[i, 1]**2 + self.kutta_normal[i, 2]**2)

	def dirichletVelocityAndPressure(self, Uinf):
		self.u = np.zeros(self.mesh.nrFaces)
		self.v = np.zeros(self.mesh.nrFaces)
		self.w = np.zeros(self.mesh.nrFaces)

		for i in range(0, self.nrStrips):
			for j in range(0, self.nrSegments):
				i0 = i*self.nrSegments + j

				iN = (i+1)*self.nrSegments + j
				iS = (i-1)*self.nrSegments + j

				iE = i*self.nrSegments + j + 1
				iW = i*self.nrSegments + j - 1

				p0 = self.mesh.face_center[i0]

				mu0 = self.doublet_strength[i0]

				if i == 0:
					pN = self.mesh.face_center[iN]

					L_l = np.sqrt((pN[0] - p0[0])**2 + (pN[1] - p0[1])**2 + (pN[2] - p0[2])**2)

					muN = self.doublet_strength[iN]

					u_l = -4*np.pi*(mu0 - muN)/L_l

				elif i == self.nrStrips - 1:
					pS = self.mesh.face_center[iS]

					L_l =  np.sqrt((pS[0] - p0[0])**2 + (pS[1] - p0[1])**2 + (pS[2] - p0[2])**2)

					muS = self.doublet_strength[iS]

					u_l = -4*np.pi*(muS - mu0)/L_l

				else:
					pN = self.mesh.face_center[iN]
					pS = self.mesh.face_center[iS]

					L_l = np.sqrt((pN[0] - p0[0])**2 + (pN[1] - p0[1])**2 + (pN[2] - p0[2])**2) + np.sqrt((pS[0] - p0[0])**2 + (pS[1] - p0[1])**2 + (pS[2] - p0[2])**2)

					muN = self.doublet_strength[iN]
					muS = self.doublet_strength[iS]

					u_l = -4*np.pi*(muS - muN)/L_l

				if j == 0:
					pE = self.mesh.face_center[iE]

					L_m = np.sqrt((pE[0] - p0[0])**2 + (pE[1] - p0[1])**2 + (pE[2] - p0[2])**2)

					muE = self.doublet_strength[iE]

					u_m = -4*np.pi*(muE - mu0)/L_m

				elif j == self.nrSegments - 1:
					pW = self.mesh.face_center[iW]

					L_m = np.sqrt((pW[0] - p0[0])**2 + (pW[1] - p0[1])**2 + (pW[2] - p0[2])**2)

					muW = self.doublet_strength[iW]

					u_m = -4*np.pi*(mu0 - muW)/L_m

				else:
					pE = self.mesh.face_center[iE]
					pW = self.mesh.face_center[iW]

					L_m = np.sqrt((pE[0] - p0[0])**2 + (pE[1] - p0[1])**2 + (pE[2] - p0[2])**2) + np.sqrt((pW[0] - p0[0])**2 + (pW[1] - p0[1])**2 + (pW[2] - p0[2])**2)

					muE = self.doublet_strength[iE]
					muW = self.doublet_strength[iW]

					u_m = -4*np.pi*(muE - muW)/L_m

				
				u_n =  4*np.pi*self.source_strength[i0]

				l = self.mesh.face_l[i0]
				m = self.mesh.face_m[i0]
				n = self.mesh.face_n[i0]

				self.u[i0] = u_l*l[0] + u_m*m[0] + u_n*n[0]
				self.v[i0] = u_l*l[1] + u_m*m[1] + u_n*n[1]
				self.w[i0] = u_l*l[2] + u_m*m[2] + u_n*n[2]

		self.u += Uinf[0]
		self.v += Uinf[1]
		self.w += Uinf[2]

		self.U = np.sqrt(self.u**2 + self.v**2 + self.w**2)

		self.Cp = 1 - self.U**2/(Uinf[0]**2 + Uinf[1]**2 + Uinf[2]**2)

	def rotate(self, rx, ry, rz, x0 = 0, y0 = 0, z0 = 0):
		# Rotate body mesh
		self.mesh.rotate(rx, ry, rz, x0 = x0, y0 = y0, z0 = z0)

		# Generate new wake geometry
		self.generateWakeGeometry()

		# Update necessary geoemtry data
		self.mesh.calculateFaceData()
		self.mesh.calculateFaceCoordinateSystem()
		self.generateKuttaGeometry()

	def exportTestGeometry(self):
		# Strip data
		strips_body = np.zeros(self.mesh.nrFaces)
		strips_wake = np.zeros(self.wake_mesh.nrFaces)
		for i in range(self.nrStrips):
			strips_body[self.stripFaces[i]]      = i
			strips_wake[self.wake_stripFaces[i]] = i

		self.mesh.addFaceData('strips', strips_body)
		self.wake_mesh.addFaceData('strips', strips_wake)

		# Doublet strength
		doublet_strength = np.zeros(self.mesh.nrFaces)

		for i in range(self.nrStrips):
			doublet_strength[self.stripFaces[i]] = self.stripFaceStrength[i]

		self.mesh.addFaceData('stripFaceStrength', doublet_strength)

		self.mesh.exportVTK('test_wing.vtp')
		self.wake_mesh.exportVTK('test_wake.vtp')

		
