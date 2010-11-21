C     SUBROUTINE COSQI(N,WSAVE)                                                 
C                                                                               
C     SUBROUTINE COSQI INITIALIZES THE ARRAY WSAVE WHICH IS USED IN             
C     BOTH COSQF AND COSQB. THE PRIME FACTORIZATION OF N TOGETHER WITH          
C     A TABULATION OF THE TRIGONOMETRIC FUNCTIONS ARE COMPUTED AND              
C     STORED IN WSAVE.                                                          
C                                                                               
C     INPUT PARAMETER                                                           
C                                                                               
C     N       THE LENGTH OF THE ARRAY TO BE TRANSFORMED.  THE METHOD            
C             IS MOST EFFICIENT WHEN N IS A PRODUCT OF SMALL PRIMES.            
C                                                                               
C     OUTPUT PARAMETER                                                          
C                                                                               
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 3*N+15.           
C             THE SAME WORK ARRAY CAN BE USED FOR BOTH COSQF AND COSQB          
C             AS LONG AS N REMAINS UNCHANGED. DIFFERENT WSAVE ARRAYS            
C             ARE REQUIRED FOR DIFFERENT VALUES OF N. THE CONTENTS OF           
C             WSAVE MUST NOT BE CHANGED BETWEEN CALLS OF COSQF OR COSQB.        
C                                                                               
      SUBROUTINE COSQI (N,WSAVE)                                                
      DIMENSION       WSAVE(*)                                                  
C                                                                               
      PIH = 0.5*PIMACH(DUM)                                                     
      DT = PIH/FLOAT(N)                                                         
      FK = 0.                                                                   
      DO 101 K=1,N                                                              
         FK = FK+1.                                                             
         WSAVE(K) = COS(FK*DT)                                                  
  101 CONTINUE                                                                  
      CALL RFFTI (N,WSAVE(N+1))                                                 
      RETURN                                                                    
      END                                                                       
