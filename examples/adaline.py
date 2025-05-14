import numpy as np
import matplotlib.pyplot as plt


def EQM(X,y,w):
    eqm = 0
    p_1,N = X.shape
    for k in range(N):
        x_k = X[:,k].reshape(p_1,1)
        u_k = (w.T@x_k)[0,0]
        d_k = float(y[k])
        eqm += (d_k-u_k)**2
    return eqm/(N*2)

def sign(u_k):
    return 1 if u_k>=0 else -1

X = np.array([
[1, 1],
[0, 1],
[0, 2],
[1, 0],
[2, 2],
[4, 1.5],
[1.5, 6],
[3, 5],
[3, 3],
[6, 4]
])

N,p = X.shape
# plt.figure(1)
# plt.scatter(X[:5,0],X[:5,1],c='teal',edgecolors='k')
# plt.scatter(X[5:,0],X[5:,1],c='orange',edgecolors='k')
# plt.xlim(-.5,6.5)
# plt.ylim(-.5,6.5)


X = X.T
X = np.vstack((
    -np.ones((1,N)),X
))

d = np.array([
1,
1,
1,
1,
1,
-1,
-1,
-1,
-1,
-1
])

#Adaline
num_max_epocas = 10000
passo_aprendizagem = 0.1
precisao_inv = 1e-4



epocas = 0
w = np.zeros((p+1,1))
w = np.random.random_sample((p+1,1))-.5


x1 = np.linspace(-2,10)
x2 = -w[1,0]/w[2,0]*x1 + w[0,0]/w[2,0]
x2 = np.nan_to_num(x2)
# line = plt.plot(x1,x2,c='k')
# line[0].remove()

EQM1 = 1
EQM2 = 0
hist_eqm = []
while epocas < num_max_epocas and abs(EQM1-EQM2)> precisao_inv:
    EQM1 = EQM(X,d,w)
    hist_eqm.append(EQM1)
    for k in range(N):
        x_k = X[:,k].reshape(p+1,1)
        u_k = (w.T@x_k)[0,0]
        d_k = float(d[k])
        e_k = d_k - u_k
        w = w + passo_aprendizagem*e_k*x_k
    # plt.pause(.1)
    # line[0].remove()
    x2 = -w[1,0]/w[2,0]*x1 + w[0,0]/w[2,0]
    x2 = np.nan_to_num(x2)
    # line = plt.plot(x1,x2,c='k')
    epocas+=1
    EQM2 = EQM(X,d,w)




# plt.pause(.1)
x2 = -w[1,0]/w[2,0]*x1 + w[0,0]/w[2,0]
x2 = np.nan_to_num(x2)
# plt.plot(x1,x2,c='g',lw=5)


# plt.figure(2)
# plt.plot(hist_eqm)
# plt.title("Curva de Aprendizagem")
# plt.xlabel("Épocas")
# plt.ylabel("EQM")
# plt.show()
