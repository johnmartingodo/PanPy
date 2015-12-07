import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/Computation')

import PanelObject
import Computation

class PanPy():
	def __init__(self, panelObjectList):
		self.panelObjects   = panelObjectList
		self.nrPanelObjects = len(self.panelObjects)

		self.Uinf   = np.array([1, 0, 0], dtype=np.double)

	def steadyStateNewmann(self):
		print('running simulation')
		# -------------------- Find out size of the complete system -------------------------------------
		self.nrCtrlPoints = 0
		
		for i in range(self.nrPanelObjects):
			if self.panelObjects[i].liftingSurface:
				self.nrCtrlPoints += self.panelObjects[i].mesh.nrFaces + self.panelObjects[i].nrStrips
			else:
				self.nrCtrlPoints += self.panelObjects[i].mesh.nrFaces

		# Global matrix system to solve, A x = b
		A = np.zeros((self.nrCtrlPoints, self.nrCtrlPoints))
		b = np.zeros(self.nrCtrlPoints)
		
		# -------------------- Build system -------------------------------------------------------------
		startTime = time.clock()
		i_start = 0
		for i in range(self.nrPanelObjects):
			# Ctrl points on the surface
			pCtrl = self.panelObjects[i].mesh.face_center
			nCtrl = self.panelObjects[i].mesh.face_n

			nrCtrlPoints = self.panelObjects[i].mesh.nrFaces

			# Add kutta ctrl points if ctrl object is a lifting surface
			if self.panelObjects[i].liftingSurface:
				pCtrl = np.append(pCtrl, self.panelObjects[i].kutta_ctrlPoints, axis=0)
				nCtrl = np.append(nCtrl, self.panelObjects[i].kutta_normal, axis=0)

				nrCtrlPoints += self.panelObjects[i].nrStrips
			
			i_stop  = i_start + nrCtrlPoints
			j_start = 0
			for j in range(self.nrPanelObjects):
				panelObject = self.panelObjects[j]
				mesh        = self.panelObjects[j].mesh
				nrFaces     = self.panelObjects[j].mesh.nrFaces

				# Number of influence panels
				nrInfluence = nrFaces

				# Add influence form doublet strips if lifting surface
				if self.panelObjects[j].liftingSurface:
					nrInfluence += self.panelObjects[j].nrStrips 

				j_stop = j_start + nrInfluence

				A_local = np.zeros((nrCtrlPoints, nrInfluence))

				A_local[0:nrCtrlPoints, 0:nrFaces] = Computation.velocityInfluence(pCtrl, nCtrl, mesh, 'source')

				# Add contribution from doublet strips if panel object is a lifting surface
				if panelObject.liftingSurface:
					wake_mesh = self.panelObjects[j].wake_mesh
					A_doublet_wake = Computation.velocityInfluence(pCtrl, nCtrl, wake_mesh, 'doublet')
					A_doublet_body = Computation.velocityInfluence(pCtrl, nCtrl, mesh, 'doublet')

					# Combine result from body and wake into strip based quantities
					for k in range(panelObject.nrStrips):
						body_stripFaces = self.panelObjects[j].stripFaces[k]          # Indices of faces in strip from body
						wake_stripFaces = self.panelObjects[j].wake_stripFaces[k]     # Indices of faces in strip from wake

						stripFaceStrength = self.panelObjects[j].stripFaceStrength[k] # strength value, relative to unknown strip value 

						A_local[0:nrCtrlPoints, nrFaces + k]  = np.sum(A_doublet_body[0:nrCtrlPoints, body_stripFaces]*stripFaceStrength, axis=1)
						A_local[0:nrCtrlPoints, nrFaces + k] += np.sum(A_doublet_wake[0:nrCtrlPoints, wake_stripFaces],                   axis=1)

				# Put resulting influence matrix into global matrix
				A[i_start:i_stop, j_start:j_stop] = A_local

				j_start += nrInfluence

			# Compute right side of linear system
			b[i_start:i_stop] = Computation.freeStreamNewmann(self.Uinf, nCtrl, nrCtrlPoints)

			i_start += nrCtrlPoints

		stopTime = time.clock()
		print('Matrix assembly time:', stopTime - startTime, 's')

		# -------------------- Solve system -------------------------------------------------------------
		startTime = time.clock()
		strength = np.linalg.solve(A, b)
		stopTime = time.clock()
		print('Linear solver time:', stopTime - startTime, 's')

		# -------------------- Transfer result ----------------------------------------------------------
		i_start = 0
		for i in range(self.nrPanelObjects):
			panelObject = self.panelObjects[i]
			i_stop = i_start + panelObject.mesh.nrFaces

			panelObject.source_strength = strength[i_start:i_stop]

			i_start += panelObject.mesh.nrFaces

			if panelObject.liftingSurface:
				for j in range(panelObject.nrStrips):
					body_stripFaces = panelObject.stripFaces[j]          # Indices of faces in strip from body
					wake_stripFaces = panelObject.wake_stripFaces[j]     # Indices of faces in strip from wake

					stripFaceStrength = panelObject.stripFaceStrength[j] # strength value, relative to unknown strip value 

					panelObject.doublet_strength[body_stripFaces] = strength[i_start + j]*stripFaceStrength

					panelObject.wake_strength[wake_stripFaces] = strength[i_start + j]

				i_start += panelObject.nrStrips

	def steadyStateDirichlet(self):
		# -------------------- Find out size of the complete system -------------------------------------
		self.nrCtrlPoints = 0
		for i in range(self.nrPanelObjects):
			self.nrCtrlPoints += self.panelObjects[i].mesh.nrFaces

		# Global matrix system to solve, A x = b
		A = np.zeros((self.nrCtrlPoints, self.nrCtrlPoints))
		b = np.zeros(self.nrCtrlPoints)
		
		# For each panel object, calculate influence matrix
		startTime = time.clock()
		i_start = 0
		for i in range(self.nrPanelObjects):
			pCtrl = self.panelObjects[i].mesh.face_center
			nCtrl = self.panelObjects[i].mesh.face_n

			nrCtrlPoints = self.panelObjects[i].mesh.nrFaces

			i_stop  = i_start + nrCtrlPoints
			j_start = 0
			for j in range(self.nrPanelObjects):
				panelObject = self.panelObjects[j]
				mesh        = panelObject.mesh
				wake_mesh   = panelObject.wake_mesh

				j_stop = j_start + mesh.nrFaces

				# Calculate influence on current ctrl points from current mesh
				A_local = Computation.potentialInfluence(pCtrl, mesh, 'doublet')
				A_wake  = Computation.potentialInfluence(pCtrl, wake_mesh, 'doublet')
						
				# Combine influence from individual panels to influence from strips
				for k in range(nrCtrlPoints):
					for l in range(panelObject.nrStrips):
						wakeSum = np.sum(A_wake[k, panelObject.wake_stripFaces[l]])

						faceIndex = panelObject.stripFaces[l, 0]
						n1 = mesh.face_n[faceIndex]
						A_local[k, faceIndex] += wakeSum*np.sign(n1[1])

						faceIndex = panelObject.stripFaces[l, -1]
						n2 = mesh.face_n[faceIndex]
						A_local[k, faceIndex] += wakeSum*np.sign(n2[1])
				
				# Put resulting influence matrix into global matrix
				A[i_start:i_stop, j_start:j_stop] = A_local

				j_start += mesh.nrFaces

			# Compute right side of linear system
			sigma = Computation.freeStreamNewmann(self.Uinf, nCtrl, nrCtrlPoints)/(4*np.pi)
			B     = Computation.potentialInfluence(pCtrl, mesh, 'source')

			b[i_start:i_stop] = -np.dot(B, sigma)
			panelObject.source_strength = sigma

			i_start += nrCtrlPoints

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

			panelObject.doublet_strength = strength[i_start:i_stop]

			j_start = 0
			for j in range(panelObject.nrStrips):
				j_stop = j_start + panelObject.wake_nrPanelsPrStrip

				faceIndex1 = panelObject.stripFaces[j, 0]
				n1 = self.panelObjects[i].mesh.face_n[faceIndex1]
				faceIndex2 = panelObject.stripFaces[j, -1]
				n2 = self.panelObjects[i].mesh.face_n[faceIndex2]

				stripStrength = panelObject.doublet_strength[faceIndex1]*np.sign(n1[1]) + panelObject.doublet_strength[faceIndex2]*np.sign(n2[1])

				panelObject.wake_strength[j_start:j_stop] = stripStrength

				j_start += panelObject.wake_nrPanelsPrStrip

			i_start += panelObject.mesh.nrFaces

	def deformWake(self):
		relaxationFactor = 0.8

		for k in range(self.nrPanelObjects):
			panelObject = self.panelObjects[k]

			if panelObject.liftingSurface:
				mesh            = panelObject.wake_mesh
				nrVertsPrStrips = panelObject.wake_nrVertsPrStrip
				nrStrips        = panelObject.nrStrips

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