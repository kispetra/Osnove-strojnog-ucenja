class IntervalError(Exception):
        pass

try:
        broj=float(input("Upiši broj:"))
        if(broj>1.0 or broj<0.0):
            raise IntervalError
        if broj>=0.9 and broj<=1.0:
            print('A')
            
        elif broj>=0.8 and broj<0.9:
            print('B')
            
        elif broj>=0.7 and broj<0.8:
            print('C')
            
        elif broj>=0.6 and broj<0.7:
            print('D')
           
        elif broj<0.6:
            print('F')
           
except IntervalError:
        print('Broj nije u intervalu.')
except ValueError:
        print('Broj nije unesen.')