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
cdef double sourcePotential(double[:] p, double[:] p0, double[:, :] verts, double [:] n, double A, int nrEdges):
	cdef int i, j, i_l, i_r
	cdef double faceLength, distance, ratio, phi
	cdef double u[3]

	cdef double p1[3]
	cdef double p2[3]
	cdef double p3[3]
	cdef double p4[3]
	cdef double p0_n[3]

	faceLength = sqrt(A)
	distance   = sqrt((p[0] - p0[0])**2 + (p[1] - p0[1])**2 + (p[2] - p0[2])**2)
	ratio      = distance/faceLength

	if ratio > 3:
		phi = -A/sqrt((p[0] - p0[0])**2 + (p[1] - p0[1])**2 + (p[2] - p0[2])**2)
	else:
		phi = 0

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

				p0_n[j] = 0.25*(p1[j] + p2[j] + p3[j] + p4[j])

			phi += -A/sqrt((p[0] - p0_n[0])**2 + (p[1] - p0_n[1])**2 + (p[2] - p0_n[2])**2)

		phi /= nrEdges

	return phi

cdef double doubletPotential(double[:] p, double[:] p0, double[:, :] verts, double A, int nrEdges):
	cdef int i, j, i_l, i_r
	cdef double faceLength, distance, ratio, phi
	cdef double u[3]

	cdef double p1[3]
	cdef double p2[3]
	cdef double p3[3]
	cdef double p4[3]
	cdef double p0_n[3]

	faceLength = sqrt(A)
	distance   = sqrt((p[0] - p0[0])**2 + (p[1] - p0[1])**2 + (p[2] - p0[2])**2)
	ratio      = distance/faceLength

	if distance == 0:
		phi = -2*M_PI
	elif ratio > 3:
		phi = -A*(p[2] - p0[2])*pow_c((p[0] - p0[0])**2 + (p[1] - p0[1])**2 + (p[2] - p0[2])**2, -1.5)
	else:
		phi = 0

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

				p0_n[j] = 0.25*(p1[j] + p2[j] + p3[j] + p4[j])

			phi += -A*(p[2] - p0[2])*pow_c((p[0] - p0_n[0])**2 + (p[1] - p0_n[1])**2 + (p[2] - p0_n[2])**2, -1.5)

		phi /= nrEdges

	return phi

cdef double[:] sourceVelocity(double[:] p, double[:] p0, double[:, :] verts, double [:] n, double A, int nrEdges):
	cdef int i, j, i_l, i_r
	cdef double faceLength, distance, ratio, C
	cdef double u[3]

	cdef double p1[3]
	cdef double p2[3]
	cdef double p3[3]
	cdef double p4[3]
	cdef double p0_n[3]

	faceLength = sqrt(A)
	distance   = sqrt((p[0] - p0[0])**2 + (p[1] - p0[1])**2 + (p[2] - p0[2])**2)
	ratio      = distance/faceLength

	if distance == 0:
		u[0] = 2*M_PI*n[0]
		u[1] = 2*M_PI*n[1]
		u[2] = 2*M_PI*n[2]
	elif ratio > 3:
		C = A*pow_c((p[0] - p0[0])**2 + (p[1] - p0[1])**2 + (p[2] - p0[2])**2, -1.5)

		u[0] = C*(p[0] - p0[0])
		u[1] = C*(p[1] - p0[1])
		u[2] = C*(p[2] - p0[2])
	else:
		u[:] = [0, 0, 0]

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

				p0_n[j] = 0.25*(p1[j] + p2[j] + p3[j] + p4[j])

			C = A*pow_c((p[0] - p0_n[0])**2 + (p[1] - p0_n[1])**2 + (p[2] - p0_n[2])**2, -1.5)

			u[0] += C*(p[0] - p0_n[0])
			u[1] += C*(p[1] - p0_n[1])
			u[2] += C*(p[2] - p0_n[2])

		u[0] /= nrEdges
		u[1] /= nrEdges
		u[2] /= nrEdges

	cdef double[:] u_view = u
	return u_view

cdef double[:] doubletVelocity(double[:] p, double[:, :] verts, int nrEdges):
	cdef int i, i2, j

	cdef double r1, r2, r1xr2_x, r1xr2_y, r1xr2_z, r1xr2, r0r1, r0r2, K
	cdef double p1[3]
	cdef double p2[3]

	cdef double u[3]
	u[:] = [0, 0, 0]

	for i in range(nrEdges):
		if i == nrEdges - 1:
			i2 = 0
		else:
			i2 = i + 1

		for j in range(3):
			p1[j] = verts[i, j]
			p2[j] = verts[i2, j]

		r1 = sqrt((p[0] - p1[0])**2 + (p[1] - p1[1])**2 + (p[2] - p1[2])**2)
		r2 = sqrt((p[0] - p2[0])**2 + (p[1] - p2[1])**2 + (p[2] - p2[2])**2)

		r1xr2_x =  (p[1] - p1[1])*(p[2] - p2[2]) - (p[2] - p1[2])*(p[1] - p2[1])
		r1xr2_y = -(p[0] - p1[0])*(p[2] - p2[2]) + (p[2] - p1[2])*(p[0] - p2[0])
		r1xr2_z =  (p[0] - p1[0])*(p[1] - p2[1]) - (p[1] - p1[1])*(p[0] - p2[0])

		r1xr2 = r1xr2_x**2 + r1xr2_y**2 + r1xr2_z**2

		r0r1 = (p2[0] - p1[0])*(p[0] - p1[0]) + (p2[1] - p1[1])*(p[1] - p1[1]) + (p2[2] - p1[2])*(p[2] - p1[2])  
		r0r2 = (p2[0] - p1[0])*(p[0] - p2[0]) + (p2[1] - p1[1])*(p[1] - p2[1]) + (p2[2] - p1[2])*(p[2] - p2[2]) 

		K =  (1/r1xr2)*(r0r1/r1 - r0r2/r2)

		u[0] += K*r1xr2_x
		u[1] += K*r1xr2_y
		u[2] += K*r1xr2_z

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
				A[i, j] = sourcePotential(ctrlMesh.face_center[i], mesh.face_center[j], verts, mesh.face_n[j], mesh.face_area[j], nrEdges)
			elif type_nr == 1:
				A[i, j] = doubletPotential(ctrlMesh.face_center[i], mesh.face_center[j], verts, mesh.face_area[j], nrEdges)
			elif type_nr == 2:
				u       = sourceVelocity(ctrlMesh.face_center[i], mesh.face_center[j], verts, mesh.face_n[j], mesh.face_area[j], nrEdges)
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
				u_temp = sourceVelocity(p[i], mesh.face_center[j], verts, mesh.face_n[j], mesh.face_area[j], nrEdges)
			if type_nr == 1:
				u_temp = doubletVelocity(p[i], verts, nrEdges)

			u[i, 0] = u[i, 0] + strength[j]*u_temp[0]
			u[i, 1] = u[i, 1] + strength[j]*u_temp[1]
			u[i, 2] = u[i, 2] + strength[j]*u_temp[2]

	return np.asarray(u)