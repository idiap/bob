C     SUBROUTINE EZFFTI(N,WSAVE)                                                
C                                                                               
C     SUBROUTINE EZFFTI INITIALIZES THE ARRAY WSAVE WHICH IS USED IN            
C     BOTH EZFFTF AND EZFFTB. THE PRIME FACTORIZATION OF N TOGETHER WITH        
C     A TABULATION OF THE TRIGONOMETRIC FUNCTIONS ARE COMPUTED AND              
C     STORED IN WSAVE.                                                          
C                                                                               
C     INPUT PARAMETER                                                           
C                                                                               
C     N       THE LENGTH OF THE SEQUENCE TO BE TRANSFORMED.                     
C                                                                               
C     OUTPUT PARAMETER                                                          
C                                                                               
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 3*N+15.           
C             THE SAME WORK ARRAY CAN BE USED FOR BOTH EZFFTF AND EZFFTB        
C             AS LONG AS N REMAINS UNCHANGED. DIFFERENT WSAVE ARRAYS            
C             ARE REQUIRED FOR DIFFERENT VALUES OF N.                           
C                                                                               
      SUBROUTINE EZFFTI (N,WSAVE)                                               
      DIMENSION       WSAVE(*)                                                  
C                                                                               
      IF (N .EQ. 1) RETURN                                                      
      CALL EZFFT1 (N,WSAVE(2*N+1),WSAVE(3*N+1))                                 
      RETURN                                                                    
      END                                                                       
