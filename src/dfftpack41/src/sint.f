C     SUBROUTINE SINT(N,X,WSAVE)                                                
C                                                                               
C     SUBROUTINE SINT COMPUTES THE DISCRETE FOURIER SINE TRANSFORM              
C     OF AN ODD SEQUENCE X(I). THE TRANSFORM IS DEFINED BELOW AT                
C     OUTPUT PARAMETER X.                                                       
C                                                                               
C     SINT IS THE UNNORMALIZED INVERSE OF ITSELF SINCE A CALL OF SINT           
C     FOLLOWED BY ANOTHER CALL OF SINT WILL MULTIPLY THE INPUT SEQUENCE         
C     X BY 2*(N+1).                                                             
C                                                                               
C     THE ARRAY WSAVE WHICH IS USED BY SUBROUTINE SINT MUST BE                  
C     INITIALIZED BY CALLING SUBROUTINE SINTI(N,WSAVE).                         
C                                                                               
C     INPUT PARAMETERS                                                          
C                                                                               
C     N       THE LENGTH OF THE SEQUENCE TO BE TRANSFORMED.  THE METHOD         
C             IS MOST EFFICIENT WHEN N+1 IS THE PRODUCT OF SMALL PRIMES.        
C                                                                               
C     X       AN ARRAY WHICH CONTAINS THE SEQUENCE TO BE TRANSFORMED            
C                                                                               
C                                                                               
C     WSAVE   A WORK ARRAY WITH DIMENSION AT LEAST INT(2.5*N+15)                
C             IN THE PROGRAM THAT CALLS SINT. THE WSAVE ARRAY MUST BE           
C             INITIALIZED BY CALLING SUBROUTINE SINTI(N,WSAVE) AND A            
C             DIFFERENT WSAVE ARRAY MUST BE USED FOR EACH DIFFERENT             
C             VALUE OF N. THIS INITIALIZATION DOES NOT HAVE TO BE               
C             REPEATED SO LONG AS N REMAINS UNCHANGED THUS SUBSEQUENT           
C             TRANSFORMS CAN BE OBTAINED FASTER THAN THE FIRST.                 
C                                                                               
C     OUTPUT PARAMETERS                                                         
C                                                                               
C     X       FOR I=1,...,N                                                     
C                                                                               
C                  X(I)= THE SUM FROM K=1 TO K=N                                
C                                                                               
C                       2*X(K)*SIN(K*I*PI/(N+1))                                
C                                                                               
C                  A CALL OF SINT FOLLOWED BY ANOTHER CALL OF                   
C                  SINT WILL MULTIPLY THE SEQUENCE X BY 2*(N+1).                
C                  HENCE SINT IS THE UNNORMALIZED INVERSE                       
C                  OF ITSELF.                                                   
C                                                                               
C     WSAVE   CONTAINS INITIALIZATION CALCULATIONS WHICH MUST NOT BE            
C             DESTROYED BETWEEN CALLS OF SINT.                                  
C                                                                               
      SUBROUTINE SINT (N,X,WSAVE)                                               
      DIMENSION       X(*)       ,WSAVE(*)                                      
C                                                                               
      NP1 = N+1                                                                 
      IW1 = N/2+1                                                               
      IW2 = IW1+NP1                                                             
      IW3 = IW2+NP1                                                             
      CALL SINT1(N,X,WSAVE,WSAVE(IW1),WSAVE(IW2),WSAVE(IW3))                    
      RETURN                                                                    
      END                                                                       
