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

cdef double[:, :] subdivideFace(double[:] p0, double[:, :] verts, int nrEdges, int level):
	cdef int i, j, k, l, i_r, i_l
	cdef double dx, xi, eta, N1, N2, N3, N4
	cdef double p1[3]
	cdef double p2[3]
	cdef double p3[3]
	cdef double p4[3]

	cdef int n        = 2**(level-1)
	cdef int nrPoints = nrEdges*n**2
	cdef double[:, :] p = np.zeros((nrPoints, 3))

	cdef int pointIndex = 0
	for i in range(nrEdges):
		if i == 0:
			i_l = nrEdges - 1
		else:
			i_l = i - 1
		if i == nrEdges - 1:
			i_r = 0
		else:
			i_r = i + 1

		for j in range(3):
			p1[j] = verts[i, j]
			p2[j] = 0.5*(verts[i, j] + verts[i_r, j])
			p3[j] = p0[j]
			p4[j] = 0.5*(verts[i, j] + verts[i_l, j])

		if level == 1:
			for j in range(3):
				p[pointIndex, j] = 0.25*(p1[j] + p2[j] + p3[j] + p4[j])
			pointIndex += 1
		else:
			dx = 2/n

			for j in range(n):
				for k in range(n):
					xi  = -1 + (0.5 + j)*dx
					eta = -1 + (0.5 + k)*dx

					N1 = 0.25*(xi - 1)*(eta - 1)
					N2 = -0.25*(xi + 1)*(eta - 1)
					N3 = 0.25*(xi + 1)*(eta + 1)
					N4 = -0.25*(xi - 1)*(eta + 1)

					for l in range(3):
						p[pointIndex, l] = N1*p1[l] + N2*p2[l] + N3*p3[l] + N4*p4[l]

					pointIndex += 1

	return p

cdef double calculateC(double PN, double RNUM, double DNOM, double Al):
	cdef double C
	cdef double sign

	sign = Al/fabs(Al)

	if PN == 0:
		if DNOM < 0:
			C = -M_PI
		elif DNOM == 0:
			C = -0.5*M_PI
		elif DNOM > 0:
			C = 0.0
	else:
		C = atan(RNUM/DNOM)

	return C
# ------------------- Face functions --------------------------------------------------------------------------
cdef double sourcePotential(double[:] p, double[:] p0, double[:, :] verts, double[:] l, double[:] m, double[:] n, int nrEdges):
	cdef int i, j, i_l, i_r
	cdef double phi, A, B, PN, SL, SM, AL, AM, Al, PA, PB, RNUM, DNOM, PL, PM, C, GL
	cdef double s[3]
	cdef double a[3]
	cdef double b[3]
	cdef double p1[3]
	cdef double p2[3]

	PN = (p[0] - p0[0])*n[0] + (p[1] - p0[1])*n[1] + (p[2] - p0[2])*n[2]

	phi = 0
	for i in range(nrEdges):
		if i == nrEdges - 1:
			i2 = 0
		else:
			i2 = i + 1

		for j in range(3):
			p1[j] = verts[i,  j] - p0[j]
			p2[j] = verts[i2, j] - p0[j]

			s[j] = p2[j] - p1[j]
			a[j] = p[j] - p0[j] - p1[j]
			b[j] = p[j] - p0[j] - p2[j]

		S = sqrt(s[0]**2 + s[1]**2 + s[2]**2)

		A = sqrt(a[0]**2 + a[1]**2 + a[2]**2)
		B = sqrt(b[0]**2 + b[1]**2 + b[2]**2)

		SL = s[0]*l[0] + s[1]*l[1] + s[2]*l[2]
		SM = s[0]*m[0] + s[1]*m[1] + s[2]*m[2]

		AL = a[0]*l[0] + a[1]*l[1] + a[2]*l[2]
		AM = a[0]*m[0] + a[1]*m[1] + a[2]*m[2]
		
		Al = AM*SL - AL*SM

		PA = (PN**2)*SL + Al*AM
		PB = PA - Al*SM

		RNUM = SM*PN*(B*PA - A*PB)
		DNOM = PA*PB + (PN**2)*A*B*(SM**2)

		GL = (1/S)*log(fabs((A+B+S)/(A+B-S)))

		C = atan2(RNUM, DNOM)

		phi += Al*GL - PN*C
	
	return phi

cdef double doubletPotential(double[:] p, double[:] p0, double[:, :] verts, double[:] l, double[:] m, double[:] n, int nrEdges):
	cdef int i, j, i_l, i_r
	cdef double phi, A, B, PN, SL, SM, AL, AM, Al, PA, PB, RNUM, DNOM, PL, PM, C, distance
	cdef double s[3]
	cdef double a[3]
	cdef double b[3]
	cdef double p1[3]
	cdef double p2[3]

	distance = sqrt((p[0] - p0[0])**2 + (p[1] - p0[1])**2 + (p[2] - p0[2])**2)

	PN = (p[0] - p0[0])*n[0] + (p[1] - p0[1])*n[1] + (p[2] - p0[2])*n[2]

	phi = 0

	if distance == 0.0:
		phi = -2*M_PI
	elif PN == 0.0:
		phi = 0
	else:
		for i in range(nrEdges):
			if i == nrEdges - 1:
				i2 = 0
			else:
				i2 = i + 1

			for j in range(3):
				p1[j] = verts[i,  j] - p0[j]
				p2[j] = verts[i2, j] - p0[j]
	
				s[j] = p2[j] - p1[j]
				a[j] = p[j] - p0[j] - p1[j]
				b[j] = p[j] - p0[j] - p2[j]

			S = sqrt(s[0]**2 + s[1]**2 + s[2]**2)

			A = sqrt(a[0]**2 + a[1]**2 + a[2]**2)
			B = sqrt(b[0]**2 + b[1]**2 + b[2]**2)

			SL = s[0]*l[0] + s[1]*l[1] + s[2]*l[2]
			SM = s[0]*m[0] + s[1]*m[1] + s[2]*m[2]

			AL = a[0]*l[0] + a[1]*l[1] + a[2]*l[2]
			AM = a[0]*m[0] + a[1]*m[1] + a[2]*m[2]
			
			Al = AM*SL - AL*SM

			PA = (PN**2)*SL + Al*AM
			PB = PA - Al*SM

			RNUM = SM*PN*(B*PA - A*PB)
			DNOM = PA*PB + (PN**2)*A*B*(SM**2)

			C = atan2(RNUM, DNOM)

			phi += C

	return phi

cdef double[:] sourceVelocity(double[:] p, double[:] p0, double[:, :] verts, double[:] l, double[:] m, double[:] n, int nrEdges):
	cdef int i, j, i_l, i_r
	cdef double phi, A, B, PN, SL, SM, AL, AM, Al, PA, PB, RNUM, DNOM, PL, PM, C, GL, distance
	cdef double s[3]
	cdef double a[3]
	cdef double b[3]
	cdef double p1[3]
	cdef double p2[3]
	cdef double u[3]

	distance = sqrt((p[0] - p0[0])**2 + (p[1] - p0[1])**2 + (p[2] - p0[2])**2)

	if distance == 0:
		u[0] = 2*M_PI*n[0]
		u[1] = 2*M_PI*n[1]
		u[2] = 2*M_PI*n[2]
	else:
		u[:] = [0.0, 0.0, 0.0]

		PN = (p[0] - p0[0])*n[0] + (p[1] - p0[1])*n[1] + (p[2] - p0[2])*n[2]

		for i in range(nrEdges):
			if i == nrEdges - 1:
				i2 = 0
			else:
				i2 = i + 1

			for j in range(3):
				p1[j] = verts[i,  j] - p0[j]
				p2[j] = verts[i2, j] - p0[j]

				s[j] = p2[j]   - p1[j]
				a[j] = p[j] - p0[j] - p1[j]
				b[j] = p[j] - p0[j] - p2[j]

			S = sqrt(s[0]**2 + s[1]**2 + s[2]**2)

			A = sqrt(a[0]**2 + a[1]**2 + a[2]**2)
			B = sqrt(b[0]**2 + b[1]**2 + b[2]**2)

			SL = s[0]*l[0] + s[1]*l[1] + s[2]*l[2]
			SM = s[0]*m[0] + s[1]*m[1] + s[2]*m[2]

			AL = a[0]*l[0] + a[1]*l[1] + a[2]*l[2]
			AM = a[0]*m[0] + a[1]*m[1] + a[2]*m[2]
		
			Al = AM*SL - AL*SM

			PA = (PN**2)*SL + Al*AM
			PB = PA - Al*SM

			RNUM = SM*PN*(B*PA - A*PB)
			DNOM = PA*PB + PN*PN*A*B*SM*SM

			GL = (1/S)*log(fabs((A+B+S)/(A+B-S)))

			if PN == 0:
				u[0] += GL*(SM*l[0] - SL*m[0])
				u[1] += GL*(SM*l[1] - SL*m[1])
				u[2] += GL*(SM*l[2] - SL*m[2])
			else:
				C = atan2(RNUM, DNOM)

				u[0] += GL*(SM*l[0] - SL*m[0]) + C*n[0]
				u[1] += GL*(SM*l[1] - SL*m[1]) + C*n[1]
				u[2] += GL*(SM*l[2] - SL*m[2]) + C*n[2]

	cdef double[:] u_view = u
	return u_view

cdef double[:] doubletVelocity(double[:] p, double[:, :] verts, int nrEdges):
	cdef int i, j, i2
	cdef double AVBx, AVBy, AVBz, AVB, A, B, S, ADB, K
	cdef double u[3]
	cdef double a[3]
	cdef double b[3]
	cdef double s[3]

	u[:] = [0, 0, 0]

	for i in range(nrEdges):
		if i == nrEdges - 1:
			i2 = 0
		else:
			i2 = i + 1

		for j in range(3):
			a[j] = p[j] - verts[i,  j]
			b[j] = p[j] - verts[i2, j]
			s[j] = verts[i2, j] - verts[i,  j]

		AVBx = a[1]*b[2] - b[1]*a[2]
		AVBy = a[2]*b[0] - b[2]*a[0]
		AVBz = a[0]*b[1] - b[0]*a[1]

		AVB = AVBx**2 + AVBy**2 + AVBz**2

		if AVB > 0:
			A = sqrt(a[0]**2 + a[1]**2 + a[2]**2)
			B = sqrt(b[0]**2 + b[1]**2 + b[2]**2)
			S = sqrt(s[0]**2 + s[1]**2 + s[2]**2)

			ADB = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

			K = (A+B)/(A*B*(A*B + ADB))

			u[0] += K*AVBx
			u[1] += K*AVBy
			u[2] += K*AVBz

	cdef double[:] u_view = u
	return u_view

# ------------------- Python accesible functions --------------------------------------------------------------
def velocityInfluence(double[:, :] p, double[:, :] n, Mesh mesh, panelType):
	cdef int nrRows, nrCols, i, j, k, i_start, nrEdges, type_nr
	if panelType == 'source':
		type_nr = 0
	elif panelType == 'doublet':
		type_nr = 1

	cdef double [:] u
	cdef double [:, :] verts

	nrRows = len(p)
	nrCols = mesh.nrFaces

	cdef double [:, :] A = np.zeros((nrRows, nrCols))

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
				u =  sourceVelocity(p[i], mesh.face_center[j], verts, mesh.face_l[j], mesh.face_m[j], mesh.face_n[j], nrEdges)
			elif type_nr == 1:
				u = doubletVelocity(p[i], verts, nrEdges)

			A[i, j] = u[0]*n[i, 0] + u[1]*n[i, 1] + u[2]*n[i, 2]

	return np.asarray(A)

def potentialInfluence(double[:, :] p, Mesh mesh, panelType):
	cdef int nrRows, nrCols, i, j, k, i_start, nrEdges, type_nr
	if panelType == 'source':
		type_nr = 0
	elif panelType == 'doublet':
		type_nr = 1

	cdef double [:, :] verts

	nrRows = len(p)
	nrCols = mesh.nrFaces

	cdef double [:, :] A = np.zeros((nrRows, nrCols))

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
				A[i, j] =  sourcePotential(p[i], mesh.face_center[j], verts, mesh.face_l[j], mesh.face_m[j], mesh.face_n[j], nrEdges)
			elif type_nr == 1:
				A[i, j] = doubletPotential(p[i], mesh.face_center[j], verts, mesh.face_l[j], mesh.face_m[j], mesh.face_n[j], nrEdges)

	return np.asarray(A)

def freeStreamNewmann(double[:] Uinf, double[:, :] n, int nrFaces):
	cdef int i

	cdef double[:] b = np.zeros(nrFaces, dtype=np.double)

	for i in range(nrFaces):
		b[i] = -(Uinf[0]*n[i, 0] + Uinf[1]*n[i, 1] + Uinf[2]*n[i, 2])
		
	return np.asarray(b)

def freeStreamPotential(double[:] Uinf, Mesh mesh):
	cdef int i

	cdef double[:] b = np.zeros(mesh.nrFaces, dtype=np.double)

	for i in range(mesh.nrFaces):
		b[i] = -(Uinf[0]*mesh.face_center[i, 0] + Uinf[1]*mesh.face_center[i, 1] + Uinf[2]*mesh.face_center[i, 2])
	
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
	cdef double[:, :] u = np.zeros((nrRows, 3), dtype=np.double)
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

			u[i, 0] += strength[j]*u_temp[0]
			u[i, 1] += strength[j]*u_temp[1]
			u[i, 2] += strength[j]*u_temp[2]

	return np.asarray(u)
