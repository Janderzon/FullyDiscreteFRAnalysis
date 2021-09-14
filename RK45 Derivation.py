import sympy as sym

#RK45 Formula 1
# B = [[],
#     [2/9],
#     [1/12, 1/4],
#     [69/128, -243/128, 135/64],
#     [-17/12, 27/4, -27/5, 16/15],
#     [65/432, -5/16, 13/16, 4/27, 5/144]]
# C = [1/9, 0, 9/20, 16/45, 1/12, 0]

#RK45 Formula 2
# B = [[],
#     [1/4],
#     [3/32, 9/32],
#     [1932/2197, -7200/2197, 7296/2197],
#     [439/216, -8, 3680/513, -845/4104],
#     [-8/27, 2, -3544/2565, 1859/4104, -11/40]]
# C = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]

#RK45 Sarafyan
B = [[],
    [1/2],
    [1/4, 1/4],
    [0, -1, 2],
    [7/27, 10/27, 0, 1/27],
    [28/625, -1/5, 546/625, 54/625, -378/625]]
C = [1/6, 0, 2/3, 1/6, 0, 0]

h = sym.Symbol('h')
y = sym.Symbol('y')
M = sym.Symbol('M')

k1 = h*y
k2 = h*(y+B[1][0]*k1)
k3 = h*(y+B[2][0]*k1+B[2][1]*k2)
k4 = h*(y+B[3][0]*k1+B[3][1]*k2+B[3][2]*k3)
k5 = h*(y+B[4][0]*k1+B[4][1]*k2+B[4][2]*k3+B[4][3]*k4)
k6 = h*(y+B[5][0]*k1+B[5][1]*k2+B[5][2]*k3+B[5][3]*k4+B[5][4]*k5)

M = 1+C[0]*k1+C[1]*k2+C[2]*k3+C[3]*k4+C[4]*k5+C[5]*k6
M = sym.simplify(M)
print(M)