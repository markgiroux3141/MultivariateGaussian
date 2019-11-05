import numpy as np
import math

class MultiVariateGaussian:
    
    @staticmethod 
    def getMulVarGaussian(x, mu, sig):
        vec_len = len(x)
        x = np.array(x)
        i_sig = np.linalg.inv(sig)
        x_mu = np.array([x - mu])
        mat_mul_1 = np.matmul(i_sig, x_mu.T)
        mat_mul_2 = np.matmul(x_mu, mat_mul_1)
        exp_res = math.exp(-0.5 * mat_mul_2)
        det_sig = np.linalg.det(sig)
        return (1/(((2*math.pi)**(vec_len/2)) * math.sqrt(det_sig))) * exp_res
    
    @staticmethod
    def getMu(vectors):
        new_vec = [0.0] * len(vectors[0])
        for i in range(len(vectors)):
            for n in range(len(vectors[0])):
                new_vec[n] += vectors[i][n]
        for i in range(len(new_vec)):
                new_vec[i] /= float(len(vectors))
        return new_vec
    
    @staticmethod
    def getSigma(vectors, mu):
        size = len(vectors[0])
        sig = np.zeros([size, size], dtype = float) 
        for i in range(len(vectors)):
            if (i % 1000) == 0:
                print("Processing element ", i)
            vec_elem = np.array(vectors[i])
            vec_elem = np.array([vec_elem - mu])
            sig_elem = np.matmul(vec_elem.T,vec_elem)
            sig += sig_elem

        sig = sig / len(vectors)
        return sig