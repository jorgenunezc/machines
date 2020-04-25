def crear_rayos(Este,Sur,Prof,T,ds,hi,xi,yi,ki,ci,thetai,ksi,ksf,ks,n,tol1):
    ''' Calculo de datos de un rayo metodo
	
	Parametros
	----------
	Sur : Coordenadas sur batimetria original
	Este : Coordenadas este batimetria original
	Prof : Profundidad en la coornada (Sur, Este)
	T : periordo xxx
	ds : 
	hi :
	xi :
	yi :
	ki :
	ci :
	thetai :
	ksi :
	ksf :
	ks :
	n : numero de pasos
	tol1 : tolerancia aceptable del modelo

	Return
	------
	rayos : Matriz(n,8) almacena datos del rayo en cada paso.
	rayos[:,0] : contiene la informacion de la coordenada x de rayo
	rayos[:,1] : contiene la informacion de la coordenada y de rayo	
	rayos[:,3] : contiene la informacion del angulo theta del rayo		
	
'''
    import numpy as np
    from scipy.interpolate import griddata
    from mis_funciones import newton
    from scipy.optimize import fsolve
    g = 9.81
    n = n # numero de rayos a determinar 
    rayos = np.zeros((n,8)) # Matriz para almacenar información
    j=0
    for i in range (1,n):
        print(i)
        contador = 1
        errori = 100
        error1 = errori
        tol1 = tol1 # Tolerancia aceptable
        while error1 > tol1 :
            contador+=1
            thetaf= thetai + ksi*ds # calculo del siguiente angulo
            theta_prom = 0.5*(thetai+thetaf) # angulo promedio
            dx =  ds*np.cos(theta_prom)
            dy =  ds*np.sin(theta_prom)
            xf = xi + dx 
            yf = yi + dy
            hf = griddata((Este, Sur), Prof,( xf, yf), method='linear')
            if hf == hi:
                cf = ci
                ksf = 0
                ks = 0.5(ksi+ksf)
                ksi = ksf
                break
            # Determinar valores de 'ki+1, Ci+1' .
            # implementar con fsolve
            def cf(k,g,T,hf):
                return g*T**2/(2*np.pi)*np.tanh(k*hf)-2*np.pi/k
            kf = fsolve(cf,ki,args=(g,T,hf))
            #cf= lambda k: g*T**2/(2*np.pi)*np.tanh(k*hf)-2*np.pi/k 
            #kf = newton(cf,ki,0.01,10**6)
            cf = 2*np.pi/(kf*T)
            c_prom = 0.5*(ci+cf)
            ksf = (np.sin(theta_prom/c_prom))*(cf-ci)/dx -(np.cos(theta_prom)/c_prom)*(cf-ci)/dy
            error1 = abs(ksf-ksi)
            #print(error1)
            ks = 0.5*(ksi+ksf)
            ksi = ksf
            if contador == 40:
                print('Te pasaste po compadre')
                break
            # termino ciclo while
            if xf<xi:
                j=1
            thetai = thetaf; xi = xf; yi = yf; ci = cf; hi = hf;
            if hi<20 :
                print('Stop ya llegamos')
                break
            rayos[i-1,:] = [xi,yi,100,thetai,ci,hi,i,j]
    return rayos
    
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
def rayos_f(g,T,k0,c0,theta0,h,xi,yi,ds,n,output):
    rayos = np.zeros((n,5))
    rayos[0,:]=[xi,yi,theta0,c0,h]
    contador = 0
    while h >= 0.1 :
        contador+=1
        if contador == n:
            break
        def cf(k,g,T,h):
            return g*T**2/(2*np.pi)*np.tanh(k*h)-2*np.pi/k
        k = fsolve(cf,k0,args=(g,T,h))
        if k*h <= np.pi/10:
            #print('aguas somera')
            c= (g*h)**0.5
        elif k*h >= np.pi :
            #print('aguas profundas')
            c = g*T/(2*np.pi)
        else :
            #print('intermedias')
            c= (g*np.tanh(k*h)/k)**0.5

        def theta_ok(theta,theta0,c,c0):
            return (theta-theta0)+((c-c0)/(2*c0))*((np.cos(2*theta))/((np.cos(theta))*(np.sin(theta))))
        theta = fsolve(theta_ok,theta0,args=(theta0,c,c0))
        dx = ds*np.cos(theta)
        dy = ds*np.sin(theta)
        xi = xi + dx
        yi = yi + dy
        h =  griddata((Este, Sur), Prof,( xi, yi), method='linear')
        #print(contador)
        theta0 = theta
        k0 = k
        c0=c
        rayos[contador,:] = [xi,yi,theta,c,h]
    print(theta)
    output.put(rayos)
    #return rayos  
#----------------------------------------------------------------------------------------------
