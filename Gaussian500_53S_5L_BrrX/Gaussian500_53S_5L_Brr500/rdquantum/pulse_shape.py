import math

def Saffman_shape(t, pulse_coe, T_gate, num_seg):
    dt = T_gate/num_seg     # Duration of each segment
    ith = int( (t - dt/2) // dt )
    ti = dt/2 + dt*ith
    ti1 = ti + dt
    
    if t < dt/2 or t > (T_gate - dt/2):
        fi = pulse_coe[0]
        fi1 = pulse_coe[0]
        ft = (fi + fi1)/2 

    else:
        if ith < 5:
            ith1 = ith + 1
            fi = pulse_coe[ith]
            fi1 = pulse_coe[ith1]
            ft = ( (fi + fi1)/2 + ( (fi1-fi)/2 ) * math.erf( (5/dt) * ( t - (ti + ti1)/2 ) ) )

        elif ith == 5:
            ith1 = ith
            fi = pulse_coe[ith]
            fi1 = pulse_coe[ith1]
            ft = ( (fi + fi1)/2 + ( (fi1-fi)/2 ) * math.erf( (5/dt) * ( t - (ti + ti1)/2 ) ) )

        elif ith > 5:
            ith1 = ith + 1
            fi = pulse_coe[num_seg-ith-1]
            fi1 = pulse_coe[num_seg-ith1-1]
            ft  = ( (fi + fi1)/2 + ( (fi1-fi)/2 ) * math.erf( (5/dt) * ( t - (ti + ti1)/2 ) ) )

    return ft