import numpy as np
import math


class material():

    def __init__(self, E11, E22, E33, v12, v13, v23, G12, G23, G13):
        self.E11 = E11
        self.E22 = E22
        self.E33 = E33
        self.v12 = v12
        self.v13 = v13
        self.v23 = v23
        self.G12 = G12
        self.G23 = G23
        self.G13 = G13
        self.layup = np.array([0,45,90,135])
        self.nplies = self.layup.size

    def plymat(self):

        v12 = self.v12
        v13 = self.v13
        v23 = self.v23

        v21 = v12*self.E22/self.E11
        v32 = v23*self.E33/self.E22
        v31 = v13*self.E33/self.E11

        self.v21 = v21
        self.v32 = v32
        self.v31 = v31


        S = 1.-v12*v21-v23*v32-v31*v13-2.*v21*v32*v13
        
        C11=(1. - v23*v32)*self.E11/S
        C12=(v21 + v31*v23)*self.E11/S
        C13=(v31 + v21*v32)*self.E11/S
        C22=(1. - v31*v13)*self.E22/S
        C23=(v32 + v31*v12)*self.E22/S
        C33=(1. - v12*v21)*self.E33/S
        C44=self.G12
        C55=self.G23
        C66=self.G13

        lawMat = np.zeros((6,6)) 
        lawMat[0,0]= C11
        lawMat[1,1]= C22
        lawMat[2,2]= C33
        lawMat[0,1]= C12
        lawMat[1,0]= C12
        lawMat[0,2]= C13
        lawMat[2,0]= C13
        lawMat[1,2]= C23
        lawMat[2,1]= C23
        lawMat[3,3]= C44 #*2.0
        lawMat[4,4]= C55 #*2.0
        lawMat[5,5]= C66 #*2.0

        return lawMat

    def sublam_mat(self):
        
        sublam = np.zeros((6,6))
        
        for degree in self.layup:
            theta = degree*np.pi/180.0
            sbar = self.lamcalc(theta)
            cmat = np.linalg.inv(sbar)
            sublam += cmat*0.25
        sublam[3,3] = sublam[3,3] #*2.0
        sublam[4,4] = sublam[4,4] #*2.0
        sublam[5,5] = sublam[5,5] #*2.0
        return sublam
    
    def lamcalc(self,theta):
        
        dtemp1 = 1.0
        dtemp2 = 1.0
        
        smat = np.zeros((6,6))
        
        v21 = self.v21
        v32 = self.v32
        v31 = self.v31
        
        v12 = v21*self.E11/self.E22
        v23 = v32*self.E22/self.E33
        v13 = v31*self.E11/self.E33
        
        self.v12 = v12
        self.v23 = v23
        self.v13 = v13
        
        smat[0,0] = 1.0/(self.E11) 
        smat[1,1] = 1.0/(self.E22) 
        smat[2,2] = 1.0/(self.E33) 
        
        smat[1,0] = -self.v12/self.E11
        smat[2,0] = -self.v13/self.E11
        smat[2,1] = -self.v23/self.E22
        
        smat[0,1] = smat[1,0]
        smat[0,2] = smat[2,0]
        smat[1,2] = smat[2,1]
        
        smat[3,3] = 1.0/(1.0*self.G12)
        smat[4,4] = 1.0/(1.0*self.G23)
        smat[5,5] = 1.0/(1.0*self.G13)
        
        R = self.transmat(theta)
        sbar = R.T @ smat @ R
        
        return sbar
    
    def transmat(self,theta):
        
        c = math.cos(theta)
        s = math.sin(theta)
        sin2x = 2*s*c
        
        R = np.zeros((6,6))
                
        R[0,0] =  c*c
        R[0,1] =  s*s
        R[0,3] =  sin2x
        R[1,0] =  s*s
        R[1,1] =  c*c
        R[1,3] =  -sin2x
        R[2,2] =  1.0
        R[4,4] =  c
        R[4,5] =  -s
        R[5,4] =  s
        R[5,5] =  c
        R[3,0] =  -s*c
        R[3,1] =  s*c
        R[3,3] =  c*c - s*s
        
        return R
    


    def materialvec(self):


        self.sublamvec = np.zeros((self.nplies,9))
        for n in range(self.nplies):
            theta = np.deg2rad(self.layup[n])
            aux1 = np.cos(theta)
            aux2 = np.sin(theta)

            R = np.array([
                [ aux1, -aux2, 0.0],
                [ aux2,  aux1, 0.0],
                [ 0.0 ,  0.0 , 1.0]
            ], dtype=np.float64)

            ef1 = R[:, 0]
            ef2 = R[:, 1]
            ef3 = np.cross(ef1, ef2)

            self.sublamvec[n, 0:3] = ef1
            self.sublamvec[n, 3:6] = ef2
            self.sublamvec[n, 6:9] = ef3


    def voigt_to_matrix(self,vec):

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


    def matrix_to_voigt(self,matrix):

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

        dSig[0] = lawMat[0, 0]*dEps[0] + lawMat[0, 1]*dEps[1] + lawMat[0, 2]*dEps[2]
        dSig[1] = lawMat[1, 0]*dEps[0] + lawMat[1, 1]*dEps[1] + lawMat[1, 2]*dEps[2]
        dSig[2] = lawMat[2, 0]*dEps[0] + lawMat[2, 1]*dEps[1] + lawMat[2, 2]*dEps[2]

        dSig[3] = lawMat[3, 3]*dEps[3]
        dSig[4] = lawMat[4, 4]*dEps[4]
        dSig[5] = lawMat[5, 5]*dEps[5]

        return dSig
    

    def sublam_sig_str(self, strainavg):


        sublamsig = np.zeros((4, 6))
        sublamstr = np.zeros((4, 6))

        eg1 = np.array([1.0, 0.0, 0.0])
        eg2 = np.array([0.0, 1.0, 0.0])
        eg3 = np.array([0.0, 0.0, 1.0])

        mat_g = self.voigt_to_matrix(strainavg)

        for n in range(4):
            ef1 = self.sublamvec[n, 0:3]
            ef2 = self.sublamvec[n, 3:6]
            ef3 = self.sublamvec[n, 6:9]

            mat_l = self.transform(eg1, eg2, eg3, mat_g, ef1, ef2, ef3)

            eps = self.matrix_to_voigt(mat_l)

            sig = self.constlaw(eps)

            sublamstr[n, :] = eps
            sublamsig[n, :] = sig

        return sublamsig, sublamstr

IM7 = material(E11=161, E22=10, E33=10, v12=0.3, v13=0.3, v23=0.43, G12=5, G13=5, G23=3.4)
plymat = IM7.plymat()
submat = IM7.sublam_mat()

IM7.materialvec()

print(IM7.nplies)

strain = np.array([0.1,0.0,0.0,0.0,0.0,0.0])
sublamsig, sublamstr = IM7.sublam_sig_str(strain)

print(sublamsig)

