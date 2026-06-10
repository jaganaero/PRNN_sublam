import math

import numpy as np


class material():

    def __init__(self, E11, E22, E33, v12, v13, v23, G12, G23, G13,
                 yt, sl, yc, ft, GIc, GIIc, Gft, CL, layup=None,
                 ply_wt=None):
        self.E11 = E11
        self.E22 = E22
        self.E33 = E33
        self.v12 = v12
        self.v13 = v13
        self.v23 = v23
        self.G12 = G12
        self.G23 = G23
        self.G13 = G13

        self.yt = yt
        self.sl = sl
        self.yc = yc
        self.ft = ft

        self.GIc = GIc
        self.GIIc = GIIc
        self.Gft = Gft

        self.CL = CL

        self.phi_0 = 53.0 * np.pi / 180.0
        self.st = self.yc / (2.0 * np.tan(self.phi_0))
        self.mut = -1.0 / np.tan(2.0 * self.phi_0)
        self.mul = (self.mut * self.sl) / self.st

        self.set_layup([0.0] if layup is None else layup, ply_wt)

        self.Dmax = 0.999999

    def set_layup(self, layup, ply_wt=None):
        self.layup = np.atleast_1d(np.asarray(layup, dtype=float))
        if self.layup.size == 0:
            raise ValueError("layup must contain at least one ply angle")

        self.nplies = self.layup.size
        if hasattr(self, "sublamvec"):
            del self.sublamvec

        if ply_wt is None:
            self.ply_wt = np.full(self.nplies, 1.0 / self.nplies, dtype=float)
            return

        self.ply_wt = np.atleast_1d(np.asarray(ply_wt, dtype=float))
        if self.ply_wt.size != self.nplies:
            raise ValueError("ply_wt must have one value per ply angle")

        wt_sum = np.sum(self.ply_wt)
        if abs(wt_sum) < 1.0e-24:
            raise ValueError("ply_wt must not sum to zero")
        self.ply_wt = self.ply_wt / wt_sum

    def plymat(self):

        v12 = self.v12
        v13 = self.v13
        v23 = self.v23

        v21 = v12 * self.E22 / self.E11
        v32 = v23 * self.E33 / self.E22
        v31 = v13 * self.E33 / self.E11

        self.v21 = v21
        self.v32 = v32
        self.v31 = v31

        S = 1.0 - v12 * v21 - v23 * v32 - v31 * v13 - 2.0 * v21 * v32 * v13

        C11 = (1.0 - v23 * v32) * self.E11 / S
        C12 = (v21 + v31 * v23) * self.E11 / S
        C13 = (v31 + v21 * v32) * self.E11 / S
        C22 = (1.0 - v31 * v13) * self.E22 / S
        C23 = (v32 + v31 * v12) * self.E22 / S
        C33 = (1.0 - v12 * v21) * self.E33 / S
        C44 = self.G12
        C55 = self.G23
        C66 = self.G13

        lawMat = np.zeros((6, 6))
        lawMat[0, 0] = C11
        lawMat[1, 1] = C22
        lawMat[2, 2] = C33
        lawMat[0, 1] = C12
        lawMat[1, 0] = C12
        lawMat[0, 2] = C13
        lawMat[2, 0] = C13
        lawMat[1, 2] = C23
        lawMat[2, 1] = C23
        lawMat[3, 3] = C44
        lawMat[4, 4] = C55
        lawMat[5, 5] = C66

        return lawMat

    def sublam_mat(self, layup=None, ply_wt=None):
        if layup is not None or ply_wt is not None:
            old_layup = self.layup.copy()
            old_ply_wt = self.ply_wt.copy()
            self.set_layup(self.layup if layup is None else layup, ply_wt)
        else:
            old_layup = None
            old_ply_wt = None

        self.plymat()
        sublam = np.zeros((6, 6))

        for n, degree in enumerate(self.layup):
            theta = degree * np.pi / 180.0
            sbar = self.lamcalc(theta)
            cmat = np.linalg.inv(sbar)
            sublam += cmat * self.ply_wt[n]

        if old_layup is not None:
            self.layup = old_layup
            self.ply_wt = old_ply_wt
            self.nplies = self.layup.size

        return sublam

    def lamcalc(self, theta):

        smat = np.zeros((6, 6))

        v12 = self.v12
        v13 = self.v13
        v23 = self.v23

        smat[0, 0] = 1.0 / self.E11
        smat[1, 1] = 1.0 / self.E22
        smat[2, 2] = 1.0 / self.E33

        smat[1, 0] = -v12 / self.E11
        smat[2, 0] = -v13 / self.E11
        smat[2, 1] = -v23 / self.E22

        smat[0, 1] = smat[1, 0]
        smat[0, 2] = smat[2, 0]
        smat[1, 2] = smat[2, 1]

        smat[3, 3] = 1.0 / self.G12
        smat[4, 4] = 1.0 / self.G23
        smat[5, 5] = 1.0 / self.G13

        R = self.transmat(theta)
        sbar = R.T @ smat @ R

        return sbar

    def transmat(self, theta):

        c = math.cos(theta)
        s = math.sin(theta)
        sin2x = 2.0 * s * c

        R = np.zeros((6, 6))

        R[0, 0] = c * c
        R[0, 1] = s * s
        R[0, 3] = sin2x
        R[1, 0] = s * s
        R[1, 1] = c * c
        R[1, 3] = -sin2x
        R[2, 2] = 1.0
        R[4, 4] = c
        R[4, 5] = -s
        R[5, 4] = s
        R[5, 5] = c
        R[3, 0] = -s * c
        R[3, 1] = s * c
        R[3, 3] = c * c - s * s

        return R

    def materialvec(self):

        self.sublamvec = np.zeros((self.nplies, 9))
        for n in range(self.nplies):
            theta = np.deg2rad(self.layup[n])
            aux1 = np.cos(theta)
            aux2 = np.sin(theta)

            R = np.array([
                [aux1, -aux2, 0.0],
                [aux2, aux1, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

            ef1 = R[:, 0]
            ef2 = R[:, 1]
            ef3 = np.cross(ef1, ef2)

            self.sublamvec[n, 0:3] = ef1
            self.sublamvec[n, 3:6] = ef2
            self.sublamvec[n, 6:9] = ef3

        return self.sublamvec

    def voigt_to_matrix(self, vec):

        matrix = np.zeros((3, 3), dtype=np.float64)

        matrix[0, 0] = vec[0]
        matrix[1, 1] = vec[1]
        matrix[2, 2] = vec[2]

        matrix[0, 1] = 0.5 * vec[3]
        matrix[1, 2] = 0.5 * vec[4]
        matrix[0, 2] = 0.5 * vec[5]

        matrix[1, 0] = 0.5 * vec[3]
        matrix[2, 1] = 0.5 * vec[4]
        matrix[2, 0] = 0.5 * vec[5]

        return matrix

    def matrix_to_voigt(self, matrix):

        vec = np.zeros(6)

        vec[0] = matrix[0, 0]
        vec[1] = matrix[1, 1]
        vec[2] = matrix[2, 2]

        vec[3] = matrix[0, 1] + matrix[1, 0]
        vec[4] = matrix[1, 2] + matrix[2, 1]
        vec[5] = matrix[0, 2] + matrix[2, 0]

        return vec

    def transform(self, a1, a2, a3, Q, b1, b2, b3):

        L = np.zeros((3, 3))

        L[0, 0] = np.dot(a1, b1)
        L[0, 1] = np.dot(a1, b2)
        L[0, 2] = np.dot(a1, b3)

        L[1, 0] = np.dot(a2, b1)
        L[1, 1] = np.dot(a2, b2)
        L[1, 2] = np.dot(a2, b3)

        L[2, 0] = np.dot(a3, b1)
        L[2, 1] = np.dot(a3, b2)
        L[2, 2] = np.dot(a3, b3)

        Qp = L.T @ Q @ L

        return Qp

    def constlaw(self, dEps):

        lawMat = self.plymat()

        dSig = np.zeros(6, dtype=np.float64)

        dSig[0] = lawMat[0, 0] * dEps[0] + lawMat[0, 1] * dEps[1] + lawMat[0, 2] * dEps[2]
        dSig[1] = lawMat[1, 0] * dEps[0] + lawMat[1, 1] * dEps[1] + lawMat[1, 2] * dEps[2]
        dSig[2] = lawMat[2, 0] * dEps[0] + lawMat[2, 1] * dEps[1] + lawMat[2, 2] * dEps[2]

        dSig[3] = lawMat[3, 3] * dEps[3]
        dSig[4] = lawMat[4, 4] * dEps[4]
        dSig[5] = lawMat[5, 5] * dEps[5]

        return dSig

    def sublam_sig_str(self, strainavg):

        if not hasattr(self, "sublamvec"):
            self.materialvec()

        sublamsig = np.zeros((self.nplies, 6))
        sublamstr = np.zeros((self.nplies, 6))

        eg1 = np.array([1.0, 0.0, 0.0])
        eg2 = np.array([0.0, 1.0, 0.0])
        eg3 = np.array([0.0, 0.0, 1.0])

        mat_g = self.voigt_to_matrix(strainavg)

        for n in range(self.nplies):
            ef1 = self.sublamvec[n, 0:3]
            ef2 = self.sublamvec[n, 3:6]
            ef3 = self.sublamvec[n, 6:9]

            mat_l = self.transform(eg1, eg2, eg3, mat_g, ef1, ef2, ef3)

            eps = self.matrix_to_voigt(mat_l)
            sig = self.constlaw(eps)

            sublamstr[n, :] = eps
            sublamsig[n, :] = sig

        return sublamsig, sublamstr

    def faildetect_sublam(self, sublamsig):

        failindex = np.zeros(2 * self.nplies, dtype=np.float64)

        for n in range(self.nplies):
            sig = sublamsig[n, 0:6]

            fi_1 = (sig[1] / self.yt) ** 2 \
                 + (sig[3] / self.sl) ** 2 \
                 + (sig[4] / self.st) ** 2

            fi_2 = (sig[4] / (self.st - self.mut * sig[1])) ** 2 \
                 + (sig[3] / (self.sl - self.mul * sig[1])) ** 2

            fi = max(abs(fi_1), abs(fi_2))

            failindex[n] = fi
            failindex[n + self.nplies] = sig[0] / self.ft

        return failindex

    def sublam_damage_init(self, sublamsig, sublamstr, failindex, cdmstat=None):

        safety = 1.0e-24

        GIc = self.GIc
        GIIc = self.GIIc
        Gft = self.Gft

        if cdmstat is None:
            cdmstat = np.zeros(8 * self.nplies, dtype=float)

        for n in range(self.nplies):
            l1 = n * 8

            if failindex[n] >= 1.0 and cdmstat[l1 + 2] != -1.0:
                sig = np.array(sublamsig[n, 0:6], dtype=float)
                epsi = np.array(sublamstr[n, 0:6], dtype=float)

                phi = 0.0 * np.pi / 180.0

                sig_R = np.zeros((3, 3), dtype=float)
                sig_R[0, 1] = sig[3] * np.cos(phi) + sig[5] * np.sin(phi)
                sig_R[1, 1] = (
                    sig[4] * np.sin(2.0 * phi)
                    + 0.5 * (sig[1] + sig[2] + np.cos(phi) * (sig[1] - sig[2]))
                )
                sig_R[1, 2] = (
                    0.5 * np.sin(2.0 * phi) * (sig[2] - sig[1])
                    + sig[4] * np.cos(2.0 * phi)
                )

                tau_omat = np.sqrt(sig_R[1, 2] ** 2 + sig_R[0, 1] ** 2)
                beta = np.arctan(sig_R[0, 1] / (sig_R[1, 2] + safety))
                omega = np.arctan(max(0.0, sig_R[1, 1]) / (tau_omat + safety))

                epsn = 0.5 * (
                    (epsi[1] + epsi[2])
                    + (epsi[1] - epsi[2]) * np.cos(2.0 * phi)
                    + epsi[4] * np.sin(2.0 * phi)
                )
                epst = (
                    -(epsi[1] - epsi[2]) * np.sin(2.0 * phi)
                    + epsi[4] * np.cos(2.0 * phi)
                )
                epsl = (sig[3] / self.G12) * np.cos(phi) + epsi[5] * np.sin(phi)

                gamma_omat = abs(epst * np.cos(beta) + epsl * np.sin(beta))

                eps0 = (
                    (max(0.0, sig_R[1, 1]) / (sig_R[1, 1] + safety)) * epsn * np.sin(omega)
                    + gamma_omat * np.cos(omega)
                )

                sig0 = np.sqrt(
                    max(0.0, sig_R[1, 1]) ** 2
                    + sig_R[1, 2] ** 2
                    + sig_R[0, 1] ** 2
                )

                Gc = (
                    ((max(0.0, sig_R[1, 1]) * max(0.0, epsn)) / (sig0 * eps0 + safety)) * GIc
                    + ((sig_R[1, 2] * epst) / (sig0 * eps0 + safety)) * GIIc
                    + ((sig_R[0, 1] * epsl) / (sig0 * eps0 + safety)) * GIIc
                )

                epsf = 2.0 * Gc / (sig0 * self.CL + safety)

                cdmstat[l1 + 0] = eps0
                cdmstat[l1 + 1] = epsf
                cdmstat[l1 + 2] = -1.0
                cdmstat[l1 + 3] = 0.0

            if failindex[n + self.nplies] >= 1.0 and cdmstat[l1 + 6] != -1.0:
                sig = np.array(sublamsig[n, 0:6], dtype=float)
                epsi = np.array(sublamstr[n, 0:6], dtype=float)

                sig0 = sig[0]
                eps0 = epsi[0]
                epsf = 2.0 * Gft / (sig0 * self.CL + safety)

                cdmstat[l1 + 4] = eps0
                cdmstat[l1 + 5] = epsf
                cdmstat[l1 + 6] = -1.0
                cdmstat[l1 + 7] = 0.0

        return cdmstat

    def sublam_damage_evol(self, sublamsig, sublamstr, cdmstat):

        safety = 1.0e-24

        for n in range(self.nplies):
            l1 = n * 8

            epsi = np.array(sublamstr[n, 0:6], dtype=float)
            sig = np.array(sublamsig[n, 0:6], dtype=float)

            if cdmstat[l1 + 2] == -1.0:

                phi = 0.0 * np.pi / 180.0

                sig_R = np.zeros((3, 3), dtype=float)
                sig_R[0, 1] = sig[3] * np.cos(phi) + sig[5] * np.sin(phi)
                sig_R[1, 1] = (
                    sig[4] * np.sin(2.0 * phi)
                    + 0.5 * (sig[1] + sig[2] + np.cos(phi) * (sig[1] - sig[2]))
                )
                sig_R[1, 2] = (
                    0.5 * np.sin(2.0 * phi) * (sig[2] - sig[1])
                    + sig[4] * np.cos(2.0 * phi)
                )

                tau_omat = np.sqrt(sig_R[1, 2] ** 2 + sig_R[0, 1] ** 2)
                beta = np.arctan(sig_R[0, 1] / (sig_R[1, 2] + safety))
                omega = np.arctan(max(0.0, sig_R[1, 1]) / (tau_omat + safety))

                epsn = 0.5 * (
                    (epsi[1] + epsi[2])
                    + (epsi[1] - epsi[2]) * np.cos(2.0 * phi)
                    + epsi[4] * np.sin(2.0 * phi)
                )
                epst = (
                    -(epsi[1] - epsi[2]) * np.sin(2.0 * phi)
                    + epsi[4] * np.cos(2.0 * phi)
                )
                epsl = (sig[3] / self.G12) * np.cos(phi) + epsi[5] * np.sin(phi)

                gamma_omat = abs(epst * np.cos(beta) + epsl * np.sin(beta))

                eps = (
                    (max(0.0, sig_R[1, 1]) / (sig_R[1, 1] + safety)) * epsn * np.sin(omega)
                    + gamma_omat * np.cos(omega)
                )

                eps0 = cdmstat[l1 + 0]
                epsf = cdmstat[l1 + 1]

                D = (epsf * (eps - eps0)) / (eps * (epsf - eps0) + safety)

                if epsf < eps0:
                    D = self.Dmax

                D = min(self.Dmax, max(0.0, D))

                cdmstat[l1 + 3] = max(D, cdmstat[l1 + 3])

            if cdmstat[l1 + 6] == -1.0:
                eps0 = cdmstat[l1 + 4]
                epsf = cdmstat[l1 + 5]
                eps = epsi[0]

                D = (epsf * (eps - eps0)) / (eps * (epsf - eps0) + safety)

                if epsf < eps0:
                    D = self.Dmax

                D = min(self.Dmax, max(0.0, D))

                cdmstat[l1 + 7] = max(D, cdmstat[l1 + 7])

        return cdmstat

    def homsig(self, sublamsig, sublamstr, cdmstat):

        if not hasattr(self, "sublamvec"):
            self.materialvec()

        eg1 = np.array([1.0, 0.0, 0.0], dtype=float)
        eg2 = np.array([0.0, 1.0, 0.0], dtype=float)
        eg3 = np.array([0.0, 0.0, 1.0], dtype=float)

        sig_tens = np.zeros((3, 3), dtype=float)
        degraded_sublamsig = np.array(sublamsig, dtype=float, copy=True)

        for n in range(self.nplies):
            l1 = n * 8

            ef1 = np.array(self.sublamvec[n, 0:3], dtype=float)
            ef2 = np.array(self.sublamvec[n, 3:6], dtype=float)
            ef3 = np.array(self.sublamvec[n, 6:9], dtype=float)

            mat_eq_stf = 1.0 - cdmstat[l1 + 3]
            fib_eq_stf = 1.0 - cdmstat[l1 + 7]

            degraded_sublamsig[n, 0] *= fib_eq_stf
            degraded_sublamsig[n, 1] *= mat_eq_stf
            degraded_sublamsig[n, 3] *= mat_eq_stf
            degraded_sublamsig[n, 4] *= mat_eq_stf

            sig_l = self.voigt_to_matrix(degraded_sublamsig[n, 0:6])
            sig_g = self.transform(ef1, ef2, ef3, sig_l, eg1, eg2, eg3)

            sig_tens = sig_tens + sig_g * self.ply_wt[n]

        sigvec = self.matrix_to_voigt(sig_tens)

        return sigvec


if __name__ == "__main__":
    IM7 = material(
        E11=161e3, E22=10e3, E33=10e3, v12=0.3, v13=0.3, v23=0.43,
        G12=5e3, G13=5e3, G23=3.4e3, yt=30.0, sl=60.0, yc=120.0,
        ft=1500.0, GIc=0.2, GIIc=1.0, Gft=195.0, CL=1.0,
        layup=[0, 45, 90, 135]
    )

    plymat = IM7.plymat()
    submat = IM7.sublam_mat()
    IM7.materialvec()

    print(IM7.nplies)

    strain = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    sublamsig, sublamstr = IM7.sublam_sig_str(strain)

    print(sublamsig)

    failindex = IM7.faildetect_sublam(sublamsig)
    cdmstat = IM7.sublam_damage_init(sublamsig, sublamstr, failindex)
    cdmstat = IM7.sublam_damage_evol(sublamsig, sublamstr, cdmstat)
    sigvec = IM7.homsig(sublamsig, sublamstr, cdmstat)

    print(failindex)
    print(sigvec)
