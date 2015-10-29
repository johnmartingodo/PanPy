#!python
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np

from Mesh import Mesh
from Mesh cimport Mesh

from libc.math cimport sqrt, atan2, atan, log, M_PI, fabs
from libc.math cimport pow as pow_c
from cython.parallel import prange

cimport cython

# ------------------- Face functions --------------------------------------------------------------------------
cdef double sourcePotential(double[:] p_g, double[:] p0, double[:, :] verts, double[:] l, double[:] m, double[:] n, int nrEdges):
	cdef int i, j, i2
	cdef double S, phi, A, B, PN, SL, SM, AL, AM, Al, PA, PB, RNUM, DNOM, C, GL, det, al, bl, am, bm
	cdef double error = 0.00000001

	cdef double s[3]
	cdef double a[3]
	cdef double b[3]
	cdef double p[3]
	cdef double p1[3]
	cdef double p2[3]

	for j in range(3):
		p[j] = p_g[j] - p0[j]

	phi = 0
	for i in range(nrEdges):
		if i == nrEdges - 1:
			i2 = 0
		else:
			i2 = i+1

		for j in range(3):
			p1[j] = verts[i, j]  - p0[j]
			p2[j] = verts[i2, j] - p0[j]

			s[j] = p2[j] - p1[j]
			a[j] = p[j]  - p1[j]
			b[j] = p[j]  - p2[j]

		S = sqrt(s[0]**2 + s[1]**2 + s[2]**2)

		if (S > 0):
			A = sqrt(a[0]**2 + a[1]**2 + a[2]**2)
			B = sqrt(b[0]**2 + b[1]**2 + b[2]**2)

			PN = p[0]*n[0] + p[1]*n[1] + p[2]*n[2]

			SL = s[0]*l[0] + s[1]*l[1] + s[2]*l[2]
			SM = s[0]*m[0] + s[1]*m[1] + s[2]*m[2]

			AL = a[0]*l[0] + a[1]*l[1] + a[2]*l[2]
			AM = a[0]*m[0] + a[1]*m[1] + a[2]*m[2]
		
			Al = AM*SL - AL*SM

			PA = PN*PN*SL + Al*AM
			PB = PA - Al*SM

			RNUM = SM*PN*(B*PA - A*PB)
			DNOM = PA*PB + PN*PN*A*B*SM*SM

			if fabs(PN) < error:
				# Find out if ctrl point is located to the left or right of line
				PL = p[0]*l[0] + p[1]*l[1] + p[2]*l[2]
				PM = p[0]*m[0] + p[1]*m[1] + p[2]*m[2]
				al = p1[0]*l[0] + p1[1]*l[1] + p1[2]*l[2]
				bl = p2[0]*l[0] + p2[1]*l[1] + p2[2]*l[2]
				am = p1[0]*m[0] + p1[1]*m[1] + p1[2]*m[2]
				bm = p2[0]*m[0] + p2[1]*m[1] + p2[2]*m[2]

				det = (al - PL)*(bm - PM) - (bl - PL)*(am - PM)

				if DNOM < 0:
					C = -M_PI                                    
				elif fabs(DNOM) < error: 
					C = -0.5*M_PI                               
				elif DNOM > 0:
					C = 0.0

				if det < 0:
					C = -C
			else:
				C = atan(RNUM/DNOM)

			GL = (1/S)*log(fabs((A+B+S)/(A+B-S)))

			phi += Al*GL - PN*C

	return phi

cdef double doubletPotential(double[:] p_g, double[:] p0, double[:, :] verts, double[:] l, double[:] m, double[:] n, int nrEdges):
	cdef int i, j
	cdef double S, phi, A, B, PN, SL, SM, AL, AM, Al, PA, PB, RNUM, DNOM, C, GL, det, al, bl, am, bm
	cdef double error = 0.00000001

	cdef double s[3]
	cdef double a[3]
	cdef double b[3]
	cdef double p[3]
	cdef double p1[3]
	cdef double p2[3]

	for j in range(3):
		p[j] = p_g[j] - p0[j]

	phi = 0
	for i in range(nrEdges):
		if i == nrEdges - 1:
			i2 = 0
		else:
			i2 = i+1

		for j in range(3):
			p1[j] = verts[i, j]  - p0[j]
			p2[j] = verts[i2, j] - p0[j]

			s[j] = p2[j] - p1[j]
			a[j] = p[j]  - p1[j]
			b[j] = p[j]  - p2[j]

		S = sqrt(s[0]**2 + s[1]**2 + s[2]**2)

		if (S > 0):
			A = sqrt(a[0]**2 + a[1]**2 + a[2]**2)
			B = sqrt(b[0]**2 + b[1]**2 + b[2]**2)

			PN = p[0]*n[0] + p[1]*n[1] + p[2]*n[2]

			SL = s[0]*l[0] + s[1]*l[1] + s[2]*l[2]
			SM = s[0]*m[0] + s[1]*m[1] + s[2]*m[2]

			AL = a[0]*l[0] + a[1]*l[1] + a[2]*l[2]
			AM = a[0]*m[0] + a[1]*m[1] + a[2]*m[2]
		
			Al = AM*SL - AL*SM

			PA = PN*PN*SL + Al*AM
			PB = PA - Al*SM

			RNUM = SM*PN*(B*PA - A*PB)
			DNOM = PA*PB + PN*PN*A*B*SM*SM

			if fabs(PN) < error:
				# Find out if ctrl point is located to the left or right of line
				PL = p[0]*l[0] + p[1]*l[1] + p[2]*l[2]
				PM = p[0]*m[0] + p[1]*m[1] + p[2]*m[2]
				al = p1[0]*l[0] + p1[1]*l[1] + p1[2]*l[2]
				bl = p2[0]*l[0] + p2[1]*l[1] + p2[2]*l[2]
				am = p1[0]*m[0] + p1[1]*m[1] + p1[2]*m[2]
				bm = p2[0]*m[0] + p2[1]*m[1] + p2[2]*m[2]

				det = (al - PL)*(bm - PM) - (bl - PL)*(am - PM)

				if DNOM < 0:
					C = M_PI                                    
				elif fabs(DNOM) < error: 
					C = 0.5*M_PI                               
				elif DNOM > 0:
					C = 0.0

				if det < 0:
					C = -C
			else:
				C = atan(RNUM/DNOM) 

			phi += C

	return phi

cdef double[:] sourceVelocity(double[:] p_g, double[:] p0, double[:, :] verts, double[:] l, double[:] m, double[:] n, int nrEdges):
	cdef int i, j
	cdef double S, phi, A, B, PL, PM, PN, SL, SM, AL, AM, Al, PA, PB, RNUM, DNOM, C, GL, det, al, bl, am, bm
	cdef double error = 0.00000001

	cdef double u[3]
	u[:] = [0, 0, 0]

	cdef double s[3]
	cdef double a[3]
	cdef double b[3]
	cdef double p[3]
	cdef double p1[3]
	cdef double p2[3]

	for j in range(3):
		p[j] = p_g[j] - p0[j]

	phi = 0
	for i in range(nrEdges):
		if i == nrEdges - 1:
			i2 = 0
		else:
			i2 = i+1

		for j in range(3):
			p1[j] = verts[i, j]  - p0[j]
			p2[j] = verts[i2, j] - p0[j]

			s[j] = p2[j] - p1[j]
			a[j] = p[j]  - p1[j]
			b[j] = p[j]  - p2[j]

		S = sqrt(s[0]**2 + s[1]**2 + s[2]**2)

		if (S > 0):
			A = sqrt(a[0]**2 + a[1]**2 + a[2]**2)
			B = sqrt(b[0]**2 + b[1]**2 + b[2]**2)

			PN = p[0]*n[0] + p[1]*n[1] + p[2]*n[2]

			SL = s[0]*l[0] + s[1]*l[1] + s[2]*l[2]
			SM = s[0]*m[0] + s[1]*m[1] + s[2]*m[2]

			AL = a[0]*l[0] + a[1]*l[1] + a[2]*l[2]
			AM = a[0]*m[0] + a[1]*m[1] + a[2]*m[2]
		
			Al = AM*SL - AL*SM

			PA = PN*PN*SL + Al*AM
			PB = PA - Al*SM

			RNUM = SM*PN*(B*PA - A*PB)
			DNOM = PA*PB + PN*PN*A*B*SM*SM

			if fabs(PN) < error:
				# Find out if ctrl point is located to the left or right of line
				PL = p[0]*l[0] + p[1]*l[1] + p[2]*l[2]
				PM = p[0]*m[0] + p[1]*m[1] + p[2]*m[2]
				al = p1[0]*l[0] + p1[1]*l[1] + p1[2]*l[2]
				bl = p2[0]*l[0] + p2[1]*l[1] + p2[2]*l[2]
				am = p1[0]*m[0] + p1[1]*m[1] + p1[2]*m[2]
				bm = p2[0]*m[0] + p2[1]*m[1] + p2[2]*m[2]

				det = (al - PL)*(bm - PM) - (bl - PL)*(am - PM)

				if DNOM < 0:
					C = M_PI                                    
				elif fabs(DNOM) < error: 
					C = 0.5*M_PI                               
				elif DNOM > 0:
					C = 0.0

				if det < 0:
					C = -C
			else:
				C = atan(RNUM/DNOM)                           
	 	
			GL = (1/S)*log(fabs((A+B+S)/(A+B-S)))

			for j in range(3):
				u[j] += GL*(SM*l[j] - SL*m[j]) + C*n[j]

	cdef double[:] u_view = u
	return u_view

cdef double[:] doubletVelocity(double[:] p, double[:, :] verts, int nrEdges):
	cdef int i, j
	cdef double AVBx, AVBy, AVBz, AVB, A, B, S, ADB, K

	cdef double s[3]
	cdef double a[3]
	cdef double b[3]
	cdef double u[3]
	u[:] = [0, 0, 0]

	phi = 0
	for i in range(nrEdges):
		if i == nrEdges - 1:
			i2 = 0
		else:
			i2 = i+1

		for j in range(3):
			s[j] = verts[i2, j] - verts[i, j] 
			a[j] = p[j]  - verts[i, j] 
			b[j] = p[j]  - verts[i2, j]

		AVBx = a[1]*b[2] - a[2]*b[1]
		AVBy = a[2]*b[0] - a[0]*b[2]
		AVBz = a[0]*b[1] - a[1]*b[0]

		AVB = AVBx**2 + AVBy**2 + AVBz**2

		A = sqrt(a[0]**2 + a[1]**2 + a[2]**2)
		B = sqrt(b[0]**2 + b[1]**2 + b[2]**2)
		S = sqrt(s[0]**2 + s[1]**2 + s[2]**2)

		if A > 0 and B > 0 and AVB > 0 and S > 0:
			ADB = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

			K = (A+B)/(A*B*(A*B + ADB))

			u[0] += K*AVBx
			u[1] += K*AVBy
			u[2] += K*AVBz

	cdef double[:] u_view = u
	return u_view

# ------------------- Python accesible functions --------------------------------------------------------------
def influenceMatrix(Mesh ctrlMesh, Mesh mesh, panelType):
	cdef int nrRows, nrCols, i, j, k, i_start, nrEdges, type_nr

	if panelType == 'sourcePotential':
		type_nr = 0
	elif panelType == 'doubletPotential':
		type_nr = 1
	elif panelType == 'sourceVelocity':
		type_nr = 2
	elif panelType == 'doubletVelocity':
		type_nr = 3

	# Initialize vectors
	cdef double[:] u

	cdef double[:, :] verts

	nrRows = ctrlMesh.nrFaces
	nrCols = mesh.nrFaces

	cdef double[:, :] A = np.zeros((nrRows, nrCols), dtype=np.double)

	for j in range(nrCols):
		nrEdges   = mesh.face_nrVerts[j]
		i_start   = mesh.face_startIndex[j]

		# Generate verts list
		verts = np.zeros((nrEdges, 3), dtype=np.double)

		for i in range(nrEdges):
			for k in range(3):
				verts[i, k] = mesh.verts[mesh.face_verts[i_start + i], k]

		for i in range(nrRows):
			if type_nr == 0:
				A[i, j] = sourcePotential(ctrlMesh.face_center[i], mesh.face_center[j], verts, mesh.face_l[j], mesh.face_m[j], mesh.face_n[j], nrEdges)
			elif type_nr == 1:
				A[i, j] = doubletPotential(ctrlMesh.face_center[i], mesh.face_center[j], verts, mesh.face_l[j], mesh.face_m[j], mesh.face_n[j], nrEdges)
			elif type_nr == 2:
				u       = sourceVelocity(ctrlMesh.face_center[i], mesh.face_center[j], verts, mesh.face_l[j], mesh.face_m[j], mesh.face_n[j], nrEdges)
				A[i, j] = u[0]*ctrlMesh.face_n[i, 0] + u[1]*ctrlMesh.face_n[i, 1] + u[2]*ctrlMesh.face_n[i, 2]
			elif type_nr == 3:
				u       = doubletVelocity(ctrlMesh.face_center[i], verts, nrEdges)
				A[i, j] = u[0]*ctrlMesh.face_n[i, 0] + u[1]*ctrlMesh.face_n[i, 1] + u[2]*ctrlMesh.face_n[i, 2]

	return np.asarray(A)

def freeStreamNewmann(double[:] Uinf, Mesh mesh):
	cdef int i

	cdef double[:] b = np.zeros(mesh.nrFaces, dtype=np.double)

	for i in range(mesh.nrFaces):
		b[i] = -(Uinf[0]*mesh.face_n[i, 0] + Uinf[1]*mesh.face_n[i, 1] + Uinf[2]*mesh.face_n[i, 2])
		
	return np.asarray(b)

def freeStreamPotential(double[:] Uinf, Mesh mesh):
	cdef int i

	cdef double[:] b = np.zeros(mesh.nrFaces, dtype=np.double)

	for i in range(mesh.nrFaces):
		b[i] = (Uinf[0]*mesh.face_center[i, 0] + Uinf[1]*mesh.face_center[i, 1] + Uinf[2]*mesh.face_center[i, 2])
	
	return np.asarray(b)

def velocity(double[:, :] p, double[:] strength, Mesh mesh, panelType):
	cdef int nrRows, nrCols, i, j, k, i_start, nrEdges, type_nr

	if panelType == 'sourceVelocity':
		type_nr = 0
	elif panelType == 'doubletVelocity':
		type_nr = 1

	nrRows = len(p)
	nrCols = mesh.nrFaces

	cdef double[:, :] verts
	cdef double[:, :] u   = np.zeros((nrRows, 3), dtype=np.double)
	cdef double[:]    u_temp 

	for j in range(nrCols):
		nrEdges   = mesh.face_nrVerts[j]
		i_start   = mesh.face_startIndex[j]

		# Generate vert list
		verts = np.zeros((nrEdges, 3), dtype=np.double)

		for i in range(nrEdges):
			for k in range(3):
				verts[i, k] = mesh.verts[mesh.face_verts[i_start + i], k]

		for i in range(nrRows):
			if type_nr == 0:
				u_temp = sourceVelocity(p[i], mesh.face_center[j], verts, mesh.face_l[j], mesh.face_m[j], mesh.face_n[j], nrEdges)
			if type_nr == 1:
				u_temp = doubletVelocity(p[i], verts, nrEdges)

			u[i, 0] = u[i, 0] + strength[j]*u_temp[0]
			u[i, 1] = u[i, 1] + strength[j]*u_temp[1]
			u[i, 2] = u[i, 2] + strength[j]*u_temp[2]

	return np.asarray(u)