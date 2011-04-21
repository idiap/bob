/**
 * @file src/cxx/ip/src/LBP8R.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief LBP8R implementation
 */

#include "ip/LBP8R.h"

namespace ip = Torch::ip;

ip::LBP8R::LBP8R(const int R, const bool to_average, 
    const bool add_average_bit, const bool uniform, 
    const bool rotation_invariant): 
  LBP(8,R,to_average,add_average_bit,uniform,rotation_invariant)
{
  // Initialize the lookup tables
  init_lut_RI();
  init_lut_U2();
  init_lut_U2RI();
  init_lut_add_average_bit();
  init_lut_normal();
  init_lut_current();
}

int ip::LBP8R::getMaxLabel() const
{
return  (m_rotation_invariant ?
            (m_uniform ? 10 : // Rotation invariant + uniform
                         37) // Rotation invariant
          : (m_uniform ? 59 : // Uniform
              (m_to_average && m_add_average_bit ? 512 : // i.e. 2^9=512 vs. 2^8=256
                                            256)       // i.e. 2^8=256)
            )
        );
}

void ip::LBP8R::init_lut_RI()
{
  m_lut_RI.resize(256);

	m_lut_RI(0) = 1;
	// binary pattern with pattern 1 in 8 bits and the rest 0s;
	m_lut_RI(1) = 2;
	m_lut_RI(2) = 2;
	m_lut_RI(4) = 2;
	m_lut_RI(8) = 2;
	m_lut_RI(16) = 2;
	m_lut_RI(32) = 2;
	m_lut_RI(64) = 2;
	m_lut_RI(128) = 2;
	// binary pattern with pattern 11 in 8 bits and the rest 0s;
	m_lut_RI(3) = 3;
	m_lut_RI(6) = 3;
	m_lut_RI(12) = 3;
	m_lut_RI(24) = 3;
	m_lut_RI(48) = 3;
	m_lut_RI(96) = 3;
	m_lut_RI(129) = 3;
	m_lut_RI(192) = 3;
	// binary pattern with pattern 101 in 8 bits and the rest 0s;
	m_lut_RI(5) = 4;
	m_lut_RI(10) = 4;
	m_lut_RI(20) = 4;
	m_lut_RI(40) = 4;
	m_lut_RI(65) = 4;
	m_lut_RI(80) = 4;
	m_lut_RI(130) = 4;
	m_lut_RI(160) = 4;
	// binary pattern with pattern 111 in 8 bits and the rest 0s;
	m_lut_RI(7) = 5;
	m_lut_RI(14) = 5;
	m_lut_RI(28) = 5;
	m_lut_RI(56) = 5;
	m_lut_RI(112) = 5;
	m_lut_RI(131) = 5;
	m_lut_RI(193) = 5;
	m_lut_RI(224) = 5;
	// binary pattern with pattern 1001 in 8 bits and the rest 0s;
	m_lut_RI(9) = 6;
	m_lut_RI(18) = 6;
	m_lut_RI(33) = 6;
	m_lut_RI(36) = 6;
	m_lut_RI(66) = 6;
	m_lut_RI(72) = 6;
	m_lut_RI(132) = 6;
	m_lut_RI(144) = 6;
	// binary pattern with pattern 1011 in 8 bits and the rest 0s;
	m_lut_RI(11) = 7;
	m_lut_RI(22) = 7;
	m_lut_RI(44) = 7;
	m_lut_RI(88) = 7;
	m_lut_RI(97) = 7;
	m_lut_RI(133) = 7;
	m_lut_RI(176) = 7;
	m_lut_RI(194) = 7;
	// binary pattern with pattern 1101 in 8 bits and the rest 0s;
	m_lut_RI(13) = 8;
	m_lut_RI(26) = 8;
	m_lut_RI(52) = 8;
	m_lut_RI(67) = 8;
	m_lut_RI(104) = 8;
	m_lut_RI(134) = 8;
	m_lut_RI(161) = 8;
	m_lut_RI(208) = 8;
	// binary pattern with pattern 1111 in 8 bits and the rest 0s;
	m_lut_RI(15) = 9;
	m_lut_RI(30) = 9;
	m_lut_RI(60) = 9;
	m_lut_RI(120) = 9;
	m_lut_RI(135) = 9;
	m_lut_RI(195) = 9;
	m_lut_RI(225) = 9;
	m_lut_RI(240) = 9;
	// binary pattern with pattern 10001 in 8 bits and the rest 0s;
	m_lut_RI(17) = 10;
	m_lut_RI(34) = 10;
	m_lut_RI(68) = 10;
	m_lut_RI(136) = 10;
	// binary pattern with pattern 10011 in 8 bits and the rest 0s;
	m_lut_RI(19) = 11;
	m_lut_RI(38) = 11;
	m_lut_RI(49) = 11;
	m_lut_RI(76) = 11;
	m_lut_RI(98) = 11;
	m_lut_RI(137) = 11;
	m_lut_RI(152) = 11;
	m_lut_RI(196) = 11;
	// binary pattern with pattern 10101 in 8 bits and the rest 0s;
	m_lut_RI(21) = 12;
	m_lut_RI(42) = 12;
	m_lut_RI(69) = 12;
	m_lut_RI(81) = 12;
	m_lut_RI(84) = 12;
	m_lut_RI(138) = 12;
	m_lut_RI(162) = 12;
	m_lut_RI(168) = 12;
	// binary pattern with pattern 10111 in 8 bits and the rest 0s;
	m_lut_RI(23) = 13;
	m_lut_RI(46) = 13;
	m_lut_RI(92) = 13;
	m_lut_RI(113) = 13;
	m_lut_RI(139) = 13;
	m_lut_RI(184) = 13;
	m_lut_RI(197) = 13;
	m_lut_RI(226) = 13;
	// binary pattern with pattern 11001 in 8 bits and the rest 0s;
	m_lut_RI(25) = 14;
	m_lut_RI(35) = 14;
	m_lut_RI(50) = 14;
	m_lut_RI(70) = 14;
	m_lut_RI(100) = 14;
	m_lut_RI(140) = 14;
	m_lut_RI(145) = 14;
	m_lut_RI(200) = 14;
	// binary pattern with pattern 11011 in 8 bits and the rest 0s;
	m_lut_RI(27) = 15;
	m_lut_RI(54) = 15;
	m_lut_RI(99) = 15;
	m_lut_RI(108) = 15;
	m_lut_RI(141) = 15;
	m_lut_RI(177) = 15;
	m_lut_RI(198) = 15;
	m_lut_RI(216) = 15;
	// binary pattern with pattern 11101 in 8 bits and the rest 0s;
	m_lut_RI(29) = 16;
	m_lut_RI(58) = 16;
	m_lut_RI(71) = 16;
	m_lut_RI(116) = 16;
	m_lut_RI(142) = 16;
	m_lut_RI(163) = 16;
	m_lut_RI(209) = 16;
	m_lut_RI(232) = 16;
	// binary pattern with pattern 11111 in 8 bits and the rest 0s;
	m_lut_RI(31) = 17;
	m_lut_RI(62) = 17;
	m_lut_RI(124) = 17;
	m_lut_RI(143) = 17;
	m_lut_RI(199) = 17;
	m_lut_RI(227) = 17;
	m_lut_RI(241) = 17;
	m_lut_RI(248) = 17;
	// binary pattern with pattern 100101 in 8 bits and the rest 0s;
	m_lut_RI(37) = 18;
	m_lut_RI(41) = 18;
	m_lut_RI(73) = 18;
	m_lut_RI(74) = 18;
	m_lut_RI(82) = 18;
	m_lut_RI(146) = 18;
	m_lut_RI(148) = 18;
	m_lut_RI(164) = 18;
	// binary pattern with pattern 100111 in 8 bits and the rest 0s;
	m_lut_RI(39) = 19;
	m_lut_RI(57) = 19;
	m_lut_RI(78) = 19;
	m_lut_RI(114) = 19;
	m_lut_RI(147) = 19;
	m_lut_RI(156) = 19;
	m_lut_RI(201) = 19;
	m_lut_RI(228) = 19;
	// binary pattern with pattern 101011 in 8 bits and the rest 0s;
	m_lut_RI(43) = 20;
	m_lut_RI(86) = 20;
	m_lut_RI(89) = 20;
	m_lut_RI(101) = 20;
	m_lut_RI(149) = 20;
	m_lut_RI(172) = 20;
	m_lut_RI(178) = 20;
	m_lut_RI(202) = 20;
	// binary pattern with pattern 101101 in 8 bits and the rest 0s;
	m_lut_RI(45) = 21;
	m_lut_RI(75) = 21;
	m_lut_RI(90) = 21;
	m_lut_RI(105) = 21;
	m_lut_RI(150) = 21;
	m_lut_RI(165) = 21;
	m_lut_RI(180) = 21;
	m_lut_RI(210) = 21;
	// binary pattern with pattern 101111 in 8 bits and the rest 0s;
	m_lut_RI(47) = 22;
	m_lut_RI(94) = 22;
	m_lut_RI(121) = 22;
	m_lut_RI(151) = 22;
	m_lut_RI(188) = 22;
	m_lut_RI(203) = 22;
	m_lut_RI(229) = 22;
	m_lut_RI(242) = 22;
	// binary pattern with pattern 110011 in 8 bits and the rest 0s;
	m_lut_RI(51) = 23;
	m_lut_RI(102) = 23;
	m_lut_RI(153) = 23;
	m_lut_RI(204) = 23;
	// binary pattern with pattern 110101 in 8 bits and the rest 0s;
	m_lut_RI(53) = 24;
	m_lut_RI(77) = 24;
	m_lut_RI(83) = 24;
	m_lut_RI(106) = 24;
	m_lut_RI(154) = 24;
	m_lut_RI(166) = 24;
	m_lut_RI(169) = 24;
	m_lut_RI(212) = 24;
	// binary pattern with pattern 110111 in 8 bits and the rest 0s;
	m_lut_RI(55) = 25;
	m_lut_RI(110) = 25;
	m_lut_RI(115) = 25;
	m_lut_RI(155) = 25;
	m_lut_RI(185) = 25;
	m_lut_RI(205) = 25;
	m_lut_RI(220) = 25;
	m_lut_RI(230) = 25;
	// binary pattern with pattern 111011 in 8 bits and the rest 0s;
	m_lut_RI(59) = 26;
	m_lut_RI(103) = 26;
	m_lut_RI(118) = 26;
	m_lut_RI(157) = 26;
	m_lut_RI(179) = 26;
	m_lut_RI(206) = 26;
	m_lut_RI(217) = 26;
	m_lut_RI(236) = 26;
	// binary pattern with pattern 111101 in 8 bits and the rest 0s;
	m_lut_RI(61) = 27;
	m_lut_RI(79) = 27;
	m_lut_RI(122) = 27;
	m_lut_RI(158) = 27;
	m_lut_RI(167) = 27;
	m_lut_RI(211) = 27;
	m_lut_RI(233) = 27;
	m_lut_RI(244) = 27;
	// binary pattern with pattern 111111 in 8 bits and the rest 0s;
	m_lut_RI(63) = 28;
	m_lut_RI(126) = 28;
	m_lut_RI(159) = 28;
	m_lut_RI(207) = 28;
	m_lut_RI(231) = 28;
	m_lut_RI(243) = 28;
	m_lut_RI(249) = 28;
	m_lut_RI(252) = 28;
	// binary pattern with pattern 1010101 in 8 bits and the rest 0s;
	m_lut_RI(85) = 29;
	m_lut_RI(170) = 29;
	// binary pattern with pattern 1010111 in 8 bits and the rest 0s;
	m_lut_RI(87) = 30;
	m_lut_RI(93) = 30;
	m_lut_RI(117) = 30;
	m_lut_RI(171) = 30;
	m_lut_RI(174) = 30;
	m_lut_RI(186) = 30;
	m_lut_RI(213) = 30;
	m_lut_RI(234) = 30;
	// binary pattern with pattern 1011011 in 8 bits and the rest 0s;
	m_lut_RI(91) = 31;
	m_lut_RI(107) = 31;
	m_lut_RI(109) = 31;
	m_lut_RI(173) = 31;
	m_lut_RI(181) = 31;
	m_lut_RI(182) = 31;
	m_lut_RI(214) = 31;
	m_lut_RI(218) = 31;
	// binary pattern with pattern 1011111 in 8 bits and the rest 0s;
	m_lut_RI(95) = 32;
	m_lut_RI(125) = 32;
	m_lut_RI(175) = 32;
	m_lut_RI(190) = 32;
	m_lut_RI(215) = 32;
	m_lut_RI(235) = 32;
	m_lut_RI(245) = 32;
	m_lut_RI(250) = 32;
	// binary pattern with pattern 1101111 in 8 bits and the rest 0s;
	m_lut_RI(111) = 33;
	m_lut_RI(123) = 33;
	m_lut_RI(183) = 33;
	m_lut_RI(189) = 33;
	m_lut_RI(219) = 33;
	m_lut_RI(222) = 33;
	m_lut_RI(237) = 33;
	m_lut_RI(246) = 33;
	// binary pattern with pattern 1110111 in 8 bits and the rest 0s;
	m_lut_RI(119) = 34;
	m_lut_RI(187) = 34;
	m_lut_RI(221) = 34;
	m_lut_RI(238) = 34;
	// binary pattern with pattern 1111111 in 8 bits and the rest 0s;
	m_lut_RI(127) = 35;
	m_lut_RI(191) = 35;
	m_lut_RI(223) = 35;
	m_lut_RI(239) = 35;
	m_lut_RI(247) = 35;
	m_lut_RI(251) = 35;
	m_lut_RI(253) = 35;
	m_lut_RI(254) = 35;
	// binary pattern with pattern 11111111 in 8 bits
	m_lut_RI(255) = 36;
}


void ip::LBP8R::init_lut_U2()
{
  m_lut_U2.resize(256);
	// A) all non uniform patterns have a label of 0.
  m_lut_U2 = 0;

	// B) LBP pattern with 0 bit to 1
	m_lut_U2(0) = 1;

	// C) LBP patterns with 1 bit to 1
	m_lut_U2(128) = 2;
	m_lut_U2(64)  = 3;
	m_lut_U2(32)  = 4;
	m_lut_U2(16)  = 5;
	m_lut_U2(8)   = 6;
	m_lut_U2(4)   = 7;
	m_lut_U2(2)   = 8;
	m_lut_U2(1)   = 9;

	// D) LBP patterns with 2 bits to 1
	m_lut_U2(128+64) = 10;
	m_lut_U2(64+32)  = 11;
	m_lut_U2(32+16)  = 12;
	m_lut_U2(16+8)   = 13;
	m_lut_U2(8+4)    = 14;
	m_lut_U2(4+2)    = 15;
	m_lut_U2(2+1)    = 16;
	m_lut_U2(1+128)  = 17;

	// E) LBP patterns with 3 bits to 1
	m_lut_U2(128+64+32) = 18;
	m_lut_U2(64+32+16)  = 19;
	m_lut_U2(32+16+8)   = 20;
	m_lut_U2(16+8+4)    = 21;
	m_lut_U2(8+4+2)     = 22;
	m_lut_U2(4+2+1)     = 23;
	m_lut_U2(2+1+128)   = 24;
	m_lut_U2(1+128+64)  = 25;

	// F) LBP patterns with 4 bits to 1
	m_lut_U2(128+64+32+16) = 26;
	m_lut_U2(64+32+16+8)   = 27;
	m_lut_U2(32+16+8+4)    = 28;
	m_lut_U2(16+8+4+2)     = 29;
	m_lut_U2(8+4+2+1)      = 30;
	m_lut_U2(4+2+1+128)    = 31;
	m_lut_U2(2+1+128+64)   = 32;
	m_lut_U2(1+128+64+32)  = 33;

	// G) LBP patterns with 5 bits to 1
	m_lut_U2(128+64+32+16+8) = 34;
	m_lut_U2(64+32+16+8+4)   = 35;
	m_lut_U2(32+16+8+4+2)    = 36;
	m_lut_U2(16+8+4+2+1)     = 37;
	m_lut_U2(8+4+2+1+128)    = 38;
	m_lut_U2(4+2+1+128+64)   = 39;
	m_lut_U2(2+1+128+64+32)  = 40;
	m_lut_U2(1+128+64+32+16) = 41;

	// H) LBP patterns with 6 bits to 1
	m_lut_U2(128+64+32+16+8+4) = 42;
	m_lut_U2(64+32+16+8+4+2)   = 43;
	m_lut_U2(32+16+8+4+2+1)    = 44;
	m_lut_U2(16+8+4+2+1+128)   = 45;
	m_lut_U2(8+4+2+1+128+64)   = 46;
	m_lut_U2(4+2+1+128+64+32)  = 47;
	m_lut_U2(2+1+128+64+32+16) = 48;
	m_lut_U2(1+128+64+32+16+8) = 49;

	// I) LBP patterns with 7 bits to 1
	m_lut_U2(128+64+32+16+8+4+2) = 50;
	m_lut_U2(64+32+16+8+4+2+1)   = 51;
	m_lut_U2(32+16+8+4+2+1+128)  = 52;
	m_lut_U2(16+8+4+2+1+128+64)  = 53;
	m_lut_U2(8+4+2+1+128+64+32)  = 54;
	m_lut_U2(4+2+1+128+64+32+16) = 55;
	m_lut_U2(2+1+128+64+32+16+8) = 56;
	m_lut_U2(1+128+64+32+16+8+4) = 57;

	// J) LBP patterns with 8 bits to 1
	m_lut_U2(128+64+32+16+8+4+2+1) = 58;
}

void ip::LBP8R::init_lut_U2RI()
{
  m_lut_U2RI.resize(256);
	// A) all non uniform patterns have a label of 0.
	// All bits are 0
  m_lut_U2RI = 0;

	m_lut_U2RI(0) = 1;

	// only one bit is 1 rest are 0's
	m_lut_U2RI(1) = 2;
	m_lut_U2RI(2) = 2;
	m_lut_U2RI(4) = 2;
	m_lut_U2RI(8) = 2;
	m_lut_U2RI(16) = 2;
	m_lut_U2RI(32) = 2;
	m_lut_U2RI(64) = 2;
	m_lut_U2RI(128) = 2;

	// only  two adjacent bits are 1 rest are 0's
	m_lut_U2RI(3) = 3;
	m_lut_U2RI(6) = 3;
	m_lut_U2RI(12) = 3;
	m_lut_U2RI(24) = 3;
	m_lut_U2RI(48) = 3;
	m_lut_U2RI(96) = 3;
	m_lut_U2RI(129) = 3;
	m_lut_U2RI(192) = 3;

	// only three adjacent bits are 1 rest are 0's
	m_lut_U2RI(7) = 4;
	m_lut_U2RI(14) = 4;
	m_lut_U2RI(28) = 4;
	m_lut_U2RI(56) = 4;
	m_lut_U2RI(112) = 4;
	m_lut_U2RI(131) = 4;
	m_lut_U2RI(193) = 4;
	m_lut_U2RI(224) = 4;


	// only four adjacent bits are 1 rest are 0's
	m_lut_U2RI(15) = 5;
	m_lut_U2RI(30) = 5;
	m_lut_U2RI(60) = 5;
	m_lut_U2RI(120) = 5;
	m_lut_U2RI(135) = 5;
	m_lut_U2RI(195) = 5;
	m_lut_U2RI(225) = 5;
	m_lut_U2RI(240) = 5;

	// only five adjacent bits are 1 rest are 0's
	m_lut_U2RI(31) = 6;
	m_lut_U2RI(62) = 6;
	m_lut_U2RI(124) = 6;
	m_lut_U2RI(143) = 6;
	m_lut_U2RI(199) = 6;
	m_lut_U2RI(227) = 6;
	m_lut_U2RI(241) = 6;
	m_lut_U2RI(248) = 6;

	// only six adjacent bits are 1 rest are 0's
	m_lut_U2RI(63) = 7;
	m_lut_U2RI(126) = 7;
	m_lut_U2RI(159) = 7;
	m_lut_U2RI(207) = 7;
	m_lut_U2RI(231) = 7;
	m_lut_U2RI(243) = 7;
	m_lut_U2RI(249) = 7;
	m_lut_U2RI(252) = 7;

	// only seven adjacent bits are 1 rest are 0's
	m_lut_U2RI(127) = 8;
	m_lut_U2RI(191) = 8;
	m_lut_U2RI(223) = 8;
	m_lut_U2RI(239) = 8;
	m_lut_U2RI(247) = 8;
	m_lut_U2RI(251) = 8;
	m_lut_U2RI(253) = 8;
	m_lut_U2RI(254) = 8;
	// eight adjacent bits are 1
	m_lut_U2RI(255) = 9;
}

void ip::LBP8R::init_lut_add_average_bit()
{
  m_lut_add_average_bit.resize(512);
  blitz::firstIndex i;
  m_lut_add_average_bit = i;
}

void ip::LBP8R::init_lut_normal()
{
  m_lut_normal.resize(256);
  blitz::firstIndex i;
  m_lut_normal = i;
}
