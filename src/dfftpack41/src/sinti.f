C     SUBROUTINE SINTI(N,WSAVE)                                                 
C                                                                               
C     SUBROUTINE SINTI INITIALIZES THE ARRAY WSAVE WHICH IS USED IN             
C     SUBROUTINE SINT. THE PRIME FACTORIZATION OF N TOGETHER WITH               
C     A TABULATION OF THE TRIGONOMETRIC FUNCTIONS ARE COMPUTED AND              
C     STORED IN WSAVE.                                                          
C                                                                               
C     INPUT PARAMETER                                                           
C                                                                               
C     N       THE LENGTH OF THE SEQUENCE TO BE TRANSFORMED.  THE METHOD         
C             IS MOST EFFICIENT WHEN N+1 IS A PRODUCT OF SMALL PRIMES.          
C                                                                               
C     OUTPUT PARAMETER                                                          
C                                                                               
C     WSAVE   A WORK ARRAY WITH AT LEAST INT(2.5*N+15) LOCATIONS.               
C             DIFFERENT WSAVE ARRAYS ARE REQUIRED FOR DIFFERENT VALUES          
C             OF N. THE CONTENTS OF WSAVE MUST NOT BE CHANGED BETWEEN           
C             CALLS OF SINT.                                                    
C                                                                               
      SUBROUTINE SINTI (N,WSAVE)                                                
      DIMENSION       WSAVE(*)                                                  
C                                                                               
      PI = PIMACH(DUM)                                                          
      IF (N .LE. 1) RETURN                                                      
      NS2 = N/2                                                                 
      NP1 = N+1                                                                 
      DT = PI/FLOAT(NP1)                                                        
      DO 101 K=1,NS2                                                            
         WSAVE(K) = 2.*SIN(K*DT)                                                
  101 CONTINUE                                                                  
      CALL RFFTI (NP1,WSAVE(NS2+1))                                             
      RETURN                                                                    
      END                                                                       
