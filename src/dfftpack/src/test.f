      PROGRAM TSTFFT
C
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C
C                       VERSION 4  APRIL 1985
C
C                         A TEST DRIVER FOR
C          A PACKAGE OF FORTRAN SUBPROGRAMS FOR THE FAST FOURIER
C           TRANSFORM OF PERIODIC AND OTHER SYMMETRIC SEQUENCES
C
C                              BY
C
C                       PAUL N SWARZTRAUBER
C
C       NATIONAL CENTER FOR ATMOSPHERIC RESEARCH  BOULDER,COLORADO 80307
C
C        WHICH IS SPONSORED BY THE NATIONAL SCIENCE FOUNDATION
C
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C
C
C             THIS PROGRAM TESTS THE PACKAGE OF FAST FOURIER
C     TRANSFORMS FOR BOTH COMPLEX AND REAL PERIODIC SEQUENCES AND
C     CERTIAN OTHER SYMMETRIC SEQUENCES THAT ARE LISTED BELOW.
C
C     1.   RFFTI     INITIALIZE  RFFTF AND RFFTB
C     2.   RFFTF     FORWARD TRANSFORM OF A REAL PERIODIC SEQUENCE
C     3.   RFFTB     BACKWARD TRANSFORM OF A REAL COEFFICIENT ARRAY
C
C     4.   EZFFTI    INITIALIZE EZFFTF AND EZFFTB
C     5.   EZFFTF    A SIMPLIFIED REAL PERIODIC FORWARD TRANSFORM
C     6.   EZFFTB    A SIMPLIFIED REAL PERIODIC BACKWARD TRANSFORM
C
C     7.   SINTI     INITIALIZE SINT
C     8.   SINT      SINE TRANSFORM OF A REAL ODD SEQUENCE
C
C     9.   COSTI     INITIALIZE COST
C     10.  COST      COSINE TRANSFORM OF A REAL EVEN SEQUENCE
C
C     11.  SINQI     INITIALIZE SINQF AND SINQB
C     12.  SINQF     FORWARD SINE TRANSFORM WITH ODD WAVE NUMBERS
C     13.  SINQB     UNNORMALIZED INVERSE OF SINQF
C
C     14.  COSQI     INITIALIZE COSQF AND COSQB
C     15.  COSQF     FORWARD COSINE TRANSFORM WITH ODD WAVE NUMBERS
C     16.  COSQB     UNNORMALIZED INVERSE OF COSQF
C
C     17.  CFFTI     INITIALIZE CFFTF AND CFFTB
C     18.  CFFTF     FORWARD TRANSFORM OF A COMPLEX PERIODIC SEQUENCE
C     19.  CFFTB     UNNORMALIZED INVERSE OF CFFTF

C     *** HACKED BY HCP FOR THE DOUBLE PREC. VERSION NOVEMEMBER 1999


C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION       ND(10)     ,X(200)     ,Y(200)     ,W(2000)    ,
     1                A(100)     ,B(100)     ,AH(100)    ,BH(100)    ,
     2                XH(200)    ,CX(200)    ,CY(200)
      DOUBLE COMPLEX         CX         ,CY
      DATA ND(1),ND(2),ND(3),ND(4),ND(5),ND(6),ND(7)/120,54,49,32,4,3,2/
      SQRT2 = SQRT(2.0D0)
      NNS = 7
      DO 157 NZ=1,NNS
         N = ND(NZ)
         MODN = MOD(N,2)
         FN = FLOAT(N)
         TFN = FN+FN
         NP1 = N+1
         NM1 = N-1
         DO 101 J=1,NP1
            X(J) = SIN(FLOAT(J)*SQRT2)
            Y(J) = X(J)
            XH(J) = X(J)
  101    CONTINUE
C
C     TEST SUBROUTINES RFFTI,RFFTF AND RFFTB
C
         CALL DFFTI (N,W)
         PI = 3.14159265358979323846D0
         DT = (PI+PI)/FN
         NS2 = (N+1)/2
         IF (NS2 .LT. 2) GO TO 104
         DO 103 K=2,NS2
            SUM1 = 0.0D0
            SUM2 = 0.0D0
            ARG = FLOAT(K-1)*DT
            DO 102 I=1,N
               ARG1 = FLOAT(I-1)*ARG
               SUM1 = SUM1+X(I)*COS(ARG1)
               SUM2 = SUM2+X(I)*SIN(ARG1)
  102       CONTINUE
            Y(2*K-2) = SUM1
            Y(2*K-1) = -SUM2
  103    CONTINUE
  104    SUM1 = 0.0D0
         SUM2 = 0.0D0
         DO 105 I=1,NM1,2
            SUM1 = SUM1+X(I)
            SUM2 = SUM2+X(I+1)
  105    CONTINUE
         IF (MODN .EQ. 1) SUM1 = SUM1+X(N)
         Y(1) = SUM1+SUM2
         IF (MODN .EQ. 0) Y(N) = SUM1-SUM2
         CALL DFFTF (N,X,W)
         RFTF = 0.0D0
         DO 106 I=1,N
            RFTF = DMAX1(RFTF,ABS(X(I)-Y(I)))
            X(I) = XH(I)
  106    CONTINUE
         RFTF = RFTF/FN
         DO 109 I=1,N
            SUM = 0.5D0*X(1)
            ARG = FLOAT(I-1)*DT
            IF (NS2 .LT. 2) GO TO 108
            DO 107 K=2,NS2
               ARG1 = FLOAT(K-1)*ARG
               SUM = SUM+X(2*K-2)*COS(ARG1)-X(2*K-1)*SIN(ARG1)
  107       CONTINUE
  108       IF (MODN .EQ. 0) SUM = SUM+.5*FLOAT((-1)**(I-1))*X(N)
            Y(I) = SUM+SUM
  109    CONTINUE
         CALL DFFTB (N,X,W)
         RFTB = 0.0D0
         DO 110 I=1,N
            RFTB = DMAX1(RFTB,ABS(X(I)-Y(I)))
            X(I) = XH(I)
            Y(I) = XH(I)
  110    CONTINUE
         CALL DFFTB (N,Y,W)
         CALL DFFTF (N,Y,W)
         CF = 1.0D0/FN
         RFTFB = 0.
         DO 111 I=1,N
            RFTFB = DMAX1(RFTFB,ABS(CF*Y(I)-X(I)))
  111    CONTINUE
C
C     TEST SUBROUTINES DSINTI AND DSINT
C
         DT = PI/FN
         DO 112 I=1,NM1
            X(I) = XH(I)
  112    CONTINUE
         DO 114 I=1,NM1
            Y(I) = 0.0D0
            ARG1 = FLOAT(I)*DT
            DO 113 K=1,NM1
               Y(I) = Y(I)+X(K)*SIN(FLOAT(K)*ARG1)
  113       CONTINUE
            Y(I) = Y(I)+Y(I)
  114    CONTINUE
         CALL DSINTI (NM1,W)
         CALL DSINT (NM1,X,W)
         CF = 0.5D0/FN
         SINTT = 0.0D0
         DO 115 I=1,NM1
            SINTT = DMAX1(SINTT,ABS(X(I)-Y(I)))
            X(I) = XH(I)
            Y(I) = X(I)
  115    CONTINUE
         SINTT = CF*SINTT
         CALL DSINT (NM1,X,W)
         CALL DSINT (NM1,X,W)
         SINTFB = 0.0D0
         DO 116 I=1,NM1
            SINTFB = DMAX1(SINTFB,ABS(CF*X(I)-Y(I)))
  116    CONTINUE
C
C     TEST SUBROUTINES COSTI AND COST
C
         DO 117 I=1,NP1
            X(I) = XH(I)
  117    CONTINUE
         DO 119 I=1,NP1
            Y(I) = 0.5D0*(X(1)+FLOAT((-1)**(I+1))*X(N+1))
            ARG = FLOAT(I-1)*DT
            DO 118 K=2,N
               Y(I) = Y(I)+X(K)*COS(FLOAT(K-1)*ARG)
  118       CONTINUE
            Y(I) = Y(I)+Y(I)
  119    CONTINUE
         CALL DCOSTI (NP1,W)
         CALL DCOST (NP1,X,W)
         COSTT = 0.0D0
         DO 120 I=1,NP1
            COSTT = DMAX1(COSTT,ABS(X(I)-Y(I)))
            X(I) = XH(I)
            Y(I) = XH(I)
  120    CONTINUE
         COSTT = CF*COSTT
         CALL DCOST (NP1,X,W)
         CALL DCOST (NP1,X,W)
         COSTFB = 0.0D0
         DO 121 I=1,NP1
            COSTFB = DMAX1(COSTFB,ABS(CF*X(I)-Y(I)))
  121    CONTINUE
C
C     TEST SUBROUTINES SINQI,SINQF AND SINQB
C
         CF = 0.25D0/FN
         DO 122 I=1,N
            Y(I) = XH(I)
  122    CONTINUE
         DT = PI/(FN+FN)
         DO 124 I=1,N
            X(I) = 0.0D0
            ARG = DT*FLOAT(I)
            DO 123 K=1,N
               X(I) = X(I)+Y(K)*SIN(FLOAT(K+K-1)*ARG)
  123       CONTINUE
            X(I) = 4.0D0*X(I)
  124    CONTINUE
         CALL DSINQI (N,W)
         CALL DSINQB (N,Y,W)
         SINQBT = 0.0D0
         DO 125 I=1,N
            SINQBT = DMAX1(SINQBT,ABS(Y(I)-X(I)))
            X(I) = XH(I)
  125    CONTINUE
         SINQBT = CF*SINQBT
         DO 127 I=1,N
            ARG = FLOAT(I+I-1)*DT
            Y(I) = 0.5D0*FLOAT((-1)**(I+1))*X(N)
            DO 126 K=1,NM1
               Y(I) = Y(I)+X(K)*SIN(FLOAT(K)*ARG)
  126       CONTINUE
            Y(I) = Y(I)+Y(I)
  127    CONTINUE
         CALL DSINQF (N,X,W)
         SINQFT = 0.0D0
         DO 128 I=1,N
            SINQFT = DMAX1(SINQFT,ABS(X(I)-Y(I)))
            Y(I) = XH(I)
            X(I) = XH(I)
  128    CONTINUE
         CALL DSINQF (N,Y,W)
         CALL DSINQB (N,Y,W)
         SINQFB = 0.0D0
         DO 129 I=1,N
            SINQFB = DMAX1(SINQFB,ABS(CF*Y(I)-X(I)))
  129    CONTINUE
C
C     TEST SUBROUTINES COSQI,COSQF AND COSQB
C
         DO 130 I=1,N
            Y(I) = XH(I)
  130    CONTINUE
         DO 132 I=1,N
            X(I) = 0.0D0
            ARG = FLOAT(I-1)*DT
            DO 131 K=1,N
               X(I) = X(I)+Y(K)*COS(FLOAT(K+K-1)*ARG)
  131       CONTINUE
            X(I) = 4.0D0*X(I)
  132    CONTINUE
         CALL DCOSQI (N,W)
         CALL DCOSQB (N,Y,W)
         COSQBT = 0.0D0
         DO 133 I=1,N
            COSQBT = DMAX1(COSQBT,ABS(X(I)-Y(I)))
            X(I) = XH(I)
  133    CONTINUE
         COSQBT = CF*COSQBT
         DO 135 I=1,N
            Y(I) = 0.5D0*X(1)
            ARG = FLOAT(I+I-1)*DT
            DO 134 K=2,N
               Y(I) = Y(I)+X(K)*COS(FLOAT(K-1)*ARG)
  134       CONTINUE
            Y(I) = Y(I)+Y(I)
  135    CONTINUE
         CALL DCOSQF (N,X,W)
         COSQFT = 0.0D0
         DO 136 I=1,N
            COSQFT = DMAX1(COSQFT,ABS(Y(I)-X(I)))
            X(I) = XH(I)
            Y(I) = XH(I)
  136    CONTINUE
         COSQFT = CF*COSQFT
         CALL DCOSQB (N,X,W)
         CALL DCOSQF (N,X,W)
         COSQFB = 0.0D0
         DO 137 I=1,N
            COSQFB = DMAX1(COSQFB,ABS(CF*X(I)-Y(I)))
  137    CONTINUE
C
C     TEST PROGRAMS EZFFTI,EZFFTF,EZFFTB
C
         CALL DZFFTI(N,W)
         DO 138 I=1,N
            X(I) = XH(I)
  138    CONTINUE
         TPI = 8.0D0*ATAN(1.0D0)
         DT = TPI/FLOAT(N)
         NS2 = (N+1)/2
         CF = 2.0D0/FLOAT(N)
         NS2M = NS2-1
         IF (NS2M .LE. 0) GO TO 141
         DO 140 K=1,NS2M
            SUM1 = 0.0D0
            SUM2 = 0.0D0
            ARG = FLOAT(K)*DT
            DO 139 I=1,N
               ARG1 = FLOAT(I-1)*ARG
               SUM1 = SUM1+X(I)*COS(ARG1)
               SUM2 = SUM2+X(I)*SIN(ARG1)
  139       CONTINUE
            A(K) = CF*SUM1
            B(K) = CF*SUM2
  140    CONTINUE
  141    NM1 = N-1
         SUM1 = 0.0D0
         SUM2 = 0.0D0
         DO 142 I=1,NM1,2
            SUM1 = SUM1+X(I)
            SUM2 = SUM2+X(I+1)
  142    CONTINUE
         IF (MODN .EQ. 1) SUM1 = SUM1+X(N)
         AZERO = 0.5D0*CF*(SUM1+SUM2)
         IF (MODN .EQ. 0) A(NS2) = 0.5D0*CF*(SUM1-SUM2)
         CALL DZFFTF (N,X,AZEROH,AH,BH,W)
         DEZF1 = ABS(AZEROH-AZERO)
         IF (MODN .EQ. 0) DEZF1 = DMAX1(DEZF1,ABS(A(NS2)-AH(NS2)))
         IF (NS2M .LE. 0) GO TO 144
         DO 143 I=1,NS2M
            DEZF1 = DMAX1(DEZF1,ABS(AH(I)-A(I)),ABS(BH(I)-B(I)))
  143    CONTINUE
  144    NS2 = N/2
         IF (MODN .EQ. 0) B(NS2) = 0.0D0
         DO 146 I=1,N
            SUM = AZERO
            ARG1 = FLOAT(I-1)*DT
            DO 145 K=1,NS2
               ARG2 = FLOAT(K)*ARG1
               SUM = SUM+A(K)*COS(ARG2)+B(K)*SIN(ARG2)
  145       CONTINUE
            X(I) = SUM
  146    CONTINUE
         CALL DZFFTB (N,Y,AZERO,A,B,W)
         DEZB1 = 0.0D0
         DO 147 I=1,N
            DEZB1 = DMAX1(DEZB1,ABS(X(I)-Y(I)))
            X(I) = XH(I)
  147    CONTINUE
         CALL DZFFTF (N,X,AZERO,A,B,W)
         CALL DZFFTB (N,Y,AZERO,A,B,W)
         DEZFB = 0.0D0
         DO 148 I=1,N
            DEZFB = DMAX1(DEZFB,ABS(X(I)-Y(I)))
  148    CONTINUE
C
C     TEST  CFFTI,CFFTF,CFFTB
C
         DO 149 I=1,N
            CX(I) = DCMPLX(COS(SQRT2*FLOAT(I)),SIN(SQRT2*FLOAT(I*I)))
  149    CONTINUE
         DT = (PI+PI)/FN
         DO 151 I=1,N
            ARG1 = -FLOAT(I-1)*DT
            CY(I) = (0.0D0,0.0D0)
            DO 150 K=1,N
               ARG2 = FLOAT(K-1)*ARG1
               CY(I) = CY(I)+DCMPLX(COS(ARG2),SIN(ARG2))*CX(K)
  150       CONTINUE
  151    CONTINUE
         CALL ZFFTI (N,W)
         CALL ZFFTF (N,CX,W)
         DCFFTF = 0.0D0
         DO 152 I=1,N
            DCFFTF = DMAX1(DCFFTF,ABS(CX(I)-CY(I)))
            CX(I) = CX(I)/FN
  152    CONTINUE
         DCFFTF = DCFFTF/FN
         DO 154 I=1,N
            ARG1 = FLOAT(I-1)*DT
            CY(I) = (0.0D0,0.0D0)
            DO 153 K=1,N
               ARG2 = FLOAT(K-1)*ARG1
               CY(I) = CY(I)+DCMPLX(COS(ARG2),SIN(ARG2))*CX(K)
  153       CONTINUE
  154    CONTINUE
         CALL ZFFTB (N,CX,W)
         DCFFTB = 0.0D0
         DO 155 I=1,N
            DCFFTB = DMAX1(DCFFTB,ABS(CX(I)-CY(I)))
            CX(I) = CY(I)
  155    CONTINUE
         CF = 1.0D0/FN
         CALL ZFFTF (N,CX,W)
         CALL ZFFTB (N,CX,W)
         DCFB = 0.0D0
         DO 156 I=1,N
            DCFB = DMAX1(DCFB,ABS(CF*CX(I)-CY(I)))
  156    CONTINUE
         WRITE (6,1001) N,RFTF,RFTB,RFTFB,SINTT,SINTFB,COSTT,COSTFB,
     1                  SINQFT,SINQBT,SINQFB,COSQFT,COSQBT,COSQFB,DEZF1,
     2                  DEZB1,DEZFB,DCFFTF,DCFFTB,DCFB
  157 CONTINUE
C
C
C
 1001 FORMAT (2H0N,I5,8H RFFTF  ,E10.3,8H RFFTB  ,E10.3,8H RFFTFB ,
     1        E10.3,8H SINT   ,E10.3,8H SINTFB ,E10.3,8H COST   ,E10.3/
     2        7X,8H COSTFB ,E10.3,8H SINQF  ,E10.3,8H SINQB  ,E10.3,
     3        8H SINQFB ,E10.3,8H COSQF  ,E10.3,8H COSQB  ,E10.3/7X,
     4        8H COSQFB ,E10.3,8H DEZF   ,E10.3,8H DEZB   ,E10.3,
     5        8H DEZFB  ,E10.3,8H CFFTF  ,E10.3,8H CFFTB  ,E10.3/
     6        7X,8H CFFTFB ,E10.3)
C
      END
