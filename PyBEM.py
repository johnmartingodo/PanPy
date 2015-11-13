import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/Computation')

import PanelObject
import Computation

class PyBEM():
	def __init__(self, panelObjectList):
		self.panelObjects   = panelObjectList
		self.nrPanelObjects = len(self.panelObjects)
		self.nrCtrlPoints   = 0

		for i in range(self.nrPanelObjects):
			self.nrCtrlPoints += self.panelObjects[i].mesh.nrFaces

		# Default settings
		self.steady = True
		self.Uinf   = np.array([1, 0, 0], dtype=np.double)

	def runSimulation(self):
		startTime = time.clock()
		print('running simulation')
		
		# Algorithm for steady simulation
		if self.steady:
			self.steadyState()

		stopTime = time.clock()

		print('Simulation time:', stopTime - startTime, 's')

	def steadyState(self):
		# Global matrix system to solve, A x = b
		A = np.zeros((self.nrCtrlPoints, self.nrCtrlPoints))
		b = np.zeros(self.nrCtrlPoints)
		
		# For each panel object, calculate influence matrix
		startTime = time.clock()
		i_start = 0
		for i in range(self.nrPanelObjects):
			ctrlPanelObject = self.panelObjects[i]
			ctrlMesh        = ctrlPanelObject.mesh

			i_stop  = i_start + ctrlMesh.nrFaces
			j_start = 0
			for j in range(self.nrPanelObjects):
				panelObject = self.panelObjects[j]
				mesh        = panelObject.mesh

				j_stop = j_start + mesh.nrFaces

				# Calculate influence on current ctrl points from current mesh
				if panelObject.liftingSurface:
					if ctrlPanelObject.boundaryType == 'dirichlet':
						A_local = Computation.influenceMatrix(ctrlMesh, mesh, 'doubletPotential')
						A_wake  = Computation.influenceMatrix(ctrlMesh, panelObject.wake_mesh, 'doubletPotential')
					elif ctrlPanelObject.boundaryType == 'newmann':
						A_local = Computation.influenceMatrix(ctrlMesh, mesh, 'doubletVelocity')
						A_wake  = Computation.influenceMatrix(ctrlMesh, panelObject.wake_mesh, 'doubletVelocity')
						
					# Combine influence from individual panels to influence from strips
					for k in range(ctrlMesh.nrFaces):
						l_start = 0
						for l in range(panelObject.wake_nrStrips):
							l_stop = l_start + panelObject.wake_nrPanelsPrStrip

							wakeSum = np.sum(A_wake[k, l_start:l_stop])

							faceIndex = panelObject.trailingFaces[l]
							n1 = mesh.face_n[faceIndex[0]]
							A_local[k, faceIndex[0]] += wakeSum*np.sign(n1[1])

							if faceIndex[1] != -1:
								n2     = mesh.face_n[faceIndex[1]]
								A_local[k, faceIndex[1]] += wakeSum*np.sign(n2[1])

							l_start += panelObject.wake_nrPanelsPrStrip
				else:
					if panelObject.boundaryType == 'dirichlet':
						A_local = Computation.influenceMatrix(ctrlMesh, mesh, 'sourcePotential')
					elif panelObject.boundaryType == 'newmann':
						A_local = Computation.influenceMatrix(ctrlMesh, mesh, 'sourceVelocity')
				
				# Put resulting influence matrix into global matrix
				A[i_start:i_stop, j_start:j_stop] = A_local

				j_start += mesh.nrFaces

			# Compute right side of linear system
			if panelObject.liftingSurface:
				if panelObject.boundaryType == 'dirichlet':
					sigma = Computation.freeStreamNewmann(self.Uinf, ctrlMesh)/(4*np.pi)
					B     = Computation.influenceMatrix(ctrlMesh, ctrlMesh, 'sourcePotential')

					b[i_start:i_stop] = -np.dot(B, sigma)
					panelObject.source_strength = sigma
				elif panelObject.boundaryType == 'newmann':
					b[i_start:i_stop] = Computation.freeStreamNewmann(self.Uinf, ctrlMesh)
			else:
				if panelObject.boundaryType == 'dirichlet':
					b[i_start:i_stop] = Computation.freeStreamPotential(self.Uinf, ctrlMesh)
				elif panelObject.boundaryType == 'newmann':
					b[i_start:i_stop] = Computation.freeStreamNewmann(self.Uinf, ctrlMesh)

			i_start += ctrlMesh.nrFaces

		stopTime = time.clock()
		print('Matrix assembly time:', stopTime - startTime, 's')

		# Solve system
		startTime = time.clock()
		strength = np.linalg.solve(A, b)
		stopTime = time.clock()
		print('Linear solver time:', stopTime - startTime, 's')

		# Transfer global strength values to individual panel objects
		i_start = 0
		for i in range(self.nrPanelObjects):
			panelObject = self.panelObjects[i]
			i_stop = i_start + panelObject.mesh.nrFaces

			if panelObject.liftingSurface:
				panelObject.doublet_strength = strength[i_start:i_stop]

				j_start = 0
				for j in range(panelObject.wake_nrStrips):
					j_stop = j_start + panelObject.wake_nrPanelsPrStrip

					faceIndex = panelObject.trailingFaces[j]
					n1 = self.panelObjects[i].mesh.face_n[faceIndex[0]]
					n2 = self.panelObjects[i].mesh.face_n[faceIndex[1]]

					if faceIndex[1] != -1:
						stripStrength = panelObject.doublet_strength[faceIndex[0]]*np.sign(n1[1]) + panelObject.doublet_strength[faceIndex[1]]*np.sign(n2[1])
					else:
						stripStrength = panelObject.doublet_strength[faceIndex[0]]

					panelObject.wake_strength[j_start:j_stop] = stripStrength

					j_start += panelObject.wake_nrPanelsPrStrip
			else:
				panelObject.source_strength = strength[i_start:i_stop]

			i_start += panelObject.mesh.nrFaces

	def deformWake(self):
		relaxationFactor = 0.8

		for k in range(self.nrPanelObjects):
			panelObject = self.panelObjects[k]

			if panelObject.liftingSurface:
				mesh            = panelObject.wake_mesh
				nrVertsPrStrips = panelObject.wake_nrVertsPrStrip
				nrStrips        = panelObject.wake_nrStrips

				for i in range(nrStrips):
					vertIndices = panelObject.getWakeVerticesInStrip(i)

					for j in range(1, nrVertsPrStrips):
						i1 = vertIndices[j-1]
						i2 = vertIndices[j]

						v1 = mesh.verts[i1]
						v2 = mesh.verts[i2]

						direction = v2 - v1
						length = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
						direction /= length

						p = 0.5*(v1 + v2)

						u = self.velocityAtPoint(p)
						uMag = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
						u /= uMag

						mesh.verts[i2] = relaxationFactor*(v1 + u*length) + (1 - relaxationFactor)*v2

	def velocityAtPoint(self, p, inducedOnly = False):
		u = np.zeros(3)

		for i in range(self.nrPanelObjects):
			panelObject = self.panelObjects[i]

			u += Computation.velocity(np.array([p]), panelObject.source_strength, panelObject.mesh, 'sourceVelocity')[0]

			if panelObject.liftingSurface:
				u += Computation.velocity(np.array([p]), panelObject.doublet_strength, panelObject.mesh, 'doubletVelocity')[0]
				u += Computation.velocity(np.array([p]), panelObject.wake_strength, panelObject.wake_mesh, 'doubletVelocity')[0]

		if not(inducedOnly):
			u += self.Uinf

		return u


	def velocityAndPressure(self):
		startTime = time.clock()
		# Calculate velocity
		for i in range(self.nrPanelObjects):
			# For each ctrl point on each mesh
			p = self.panelObjects[i].mesh.face_center
			u = np.zeros((self.panelObjects[i].mesh.nrFaces, 3), dtype=np.double)

			for j in range(self.nrPanelObjects):
				# Calculate influence from all panel objects
				mesh = self.panelObjects[j].mesh

				strength = self.panelObjects[j].source_strength
				u += Computation.velocity(p, strength, mesh, 'sourceVelocity')

				if self.panelObjects[j].liftingSurface:
					strength      = self.panelObjects[j].doublet_strength
					wake_strength = self.panelObjects[j].wake_strength
					wake_mesh     = self.panelObjects[j].wake_mesh
					u += Computation.velocity(p, strength, mesh, 'doubletVelocity')
					u += Computation.velocity(p, wake_strength, wake_mesh, 'doubletVelocity') 

			self.panelObjects[i].u = u[:, 0] + self.Uinf[0]
			self.panelObjects[i].v = u[:, 1] + self.Uinf[1]
			self.panelObjects[i].w = u[:, 2] + self.Uinf[2]

			self.panelObjects[i].U = np.sqrt(self.panelObjects[i].u**2 + self.panelObjects[i].v**2 + self.panelObjects[i].w**2)

			UinfMag = np.sqrt(np.sum(self.Uinf**2))
			self.panelObjects[i].Cp = 1 - self.panelObjects[i].U**2/UinfMag**2

		stopTime = time.clock()
		print('Velocity and pressure calculation time:', stopTime - startTime, 's')

	def writeResults(self):
		for i in range(self.nrPanelObjects):
			mesh = self.panelObjects[i].mesh

			mesh.addFaceData('source_strength', self.panelObjects[i].source_strength)
			mesh.addFaceData('doublet_strength', self.panelObjects[i].doublet_strength)
			mesh.addFaceData('Cp', self.panelObjects[i].Cp)
			mesh.addFaceData('u', self.panelObjects[i].u)
			mesh.addFaceData('v', self.panelObjects[i].v)
			mesh.addFaceData('w', self.panelObjects[i].w)
			mesh.addFaceData('U', self.panelObjects[i].U)

			mesh.exportVTK('panelObject{:.0f}.vtp'.format(i))

			if self.panelObjects[i].liftingSurface:
				mesh = self.panelObjects[i].wake_mesh
				mesh.addFaceData('doublet_strength', self.panelObjects[i].wake_strength)

				mesh.exportVTK('wakeObject{:.0f}.vtp'.format(i))