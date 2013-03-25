/**
 * @file ip/cxx/LBP8R.cc
 * @date Wed Apr 20 21:44:36 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief LBP8R implementation
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/make_shared.hpp>
#include <bob/ip/LBP8R.h>

bob::ip::LBP8R::LBP8R(const double R,
    const bool circular,
    const bool to_average,
    const bool add_average_bit,
    const bool uniform, 
    const bool rotation_invariant,
    const int eLBP_type): 
  LBP(8,R,R,circular,to_average,add_average_bit,uniform,rotation_invariant,eLBP_type)
{
  init_luts();
}


bob::ip::LBP8R::LBP8R(const double R,
    const double R2,
    const bool circular,
    const bool to_average,
    const bool add_average_bit,
    const bool uniform, 
    const bool rotation_invariant,
    const int eLBP_type): 
  LBP(8,R,R2,circular,to_average,add_average_bit,uniform,rotation_invariant,eLBP_type)
{
  init_luts();
}


bob::ip::LBP8R::LBP8R(const bob::ip::LBP8R& other):
  bob::ip::LBP(other)
{
  init_luts();
}

bob::ip::LBP8R::~LBP8R() { }

bob::ip::LBP8R& bob::ip::LBP8R::operator= (const bob::ip::LBP8R& other) {
  bob::ip::LBP::operator=(other);
  return *this;
}

boost::shared_ptr<bob::ip::LBP> bob::ip::LBP8R::clone() const {
  return boost::make_shared<bob::ip::LBP8R>(*this);
}

int bob::ip::LBP8R::getMaxLabel() const
{
return  (m_rotation_invariant ?
            (m_uniform ? 10 : // Rotation invariant + uniform
                         36) // Rotation invariant
          : (m_uniform ? 59 : // Uniform
              (m_to_average && m_add_average_bit ? 512 : // i.e. 2^9=512 vs. 2^8=256
                                            256)       // i.e. 2^8=256)
            )
        );
}

void bob::ip::LBP8R::init_lut_RI()
{
  m_lut_RI.resize(256);

  m_lut_RI(0) = 0;
  // binary pattern with pattern 1 in 8 bits and the rest 0s;
  m_lut_RI(1) = 1;
  m_lut_RI(2) = 1;
  m_lut_RI(4) = 1;
  m_lut_RI(8) = 1;
  m_lut_RI(16) = 1;
  m_lut_RI(32) = 1;
  m_lut_RI(64) = 1;
  m_lut_RI(128) = 1;
  // binary pattern with pattern 11 in 8 bits and the rest 0s;
  m_lut_RI(3) = 2;
  m_lut_RI(6) = 2;
  m_lut_RI(12) = 2;
  m_lut_RI(24) = 2;
  m_lut_RI(48) = 2;
  m_lut_RI(96) = 2;
  m_lut_RI(129) = 2;
  m_lut_RI(192) = 2;
  // binary pattern with pattern 101 in 8 bits and the rest 0s;
  m_lut_RI(5) = 3;
  m_lut_RI(10) = 3;
  m_lut_RI(20) = 3;
  m_lut_RI(40) = 3;
  m_lut_RI(65) = 3;
  m_lut_RI(80) = 3;
  m_lut_RI(130) = 3;
  m_lut_RI(160) = 3;
  // binary pattern with pattern 111 in 8 bits and the rest 0s;
  m_lut_RI(7) = 4;
  m_lut_RI(14) = 4;
  m_lut_RI(28) = 4;
  m_lut_RI(56) = 4;
  m_lut_RI(112) = 4;
  m_lut_RI(131) = 4;
  m_lut_RI(193) = 4;
  m_lut_RI(224) = 4;
  // binary pattern with pattern 1001 in 8 bits and the rest 0s;
  m_lut_RI(9) = 5;
  m_lut_RI(18) = 5;
  m_lut_RI(33) = 5;
  m_lut_RI(36) = 5;
  m_lut_RI(66) = 5;
  m_lut_RI(72) = 5;
  m_lut_RI(132) = 5;
  m_lut_RI(144) = 5;
  // binary pattern with pattern 1011 in 8 bits and the rest 0s;
  m_lut_RI(11) = 6;
  m_lut_RI(22) = 6;
  m_lut_RI(44) = 6;
  m_lut_RI(88) = 6;
  m_lut_RI(97) = 6;
  m_lut_RI(133) = 6;
  m_lut_RI(176) = 6;
  m_lut_RI(194) = 6;
  // binary pattern with pattern 1101 in 8 bits and the rest 0s;
  m_lut_RI(13) = 7;
  m_lut_RI(26) = 7;
  m_lut_RI(52) = 7;
  m_lut_RI(67) = 7;
  m_lut_RI(104) = 7;
  m_lut_RI(134) = 7;
  m_lut_RI(161) = 7;
  m_lut_RI(208) = 7;
  // binary pattern with pattern 1111 in 8 bits and the rest 0s;
  m_lut_RI(15) = 8;
  m_lut_RI(30) = 8;
  m_lut_RI(60) = 8;
  m_lut_RI(120) = 8;
  m_lut_RI(135) = 8;
  m_lut_RI(195) = 8;
  m_lut_RI(225) = 8;
  m_lut_RI(240) = 8;
  // binary pattern with pattern 10001 in 8 bits and the rest 0s;
  m_lut_RI(17) = 9;
  m_lut_RI(34) = 9;
  m_lut_RI(68) = 9;
  m_lut_RI(136) = 9;
  // binary pattern with pattern 10011 in 8 bits and the rest 0s;
  m_lut_RI(19) = 10;
  m_lut_RI(38) = 10;
  m_lut_RI(49) = 10;
  m_lut_RI(76) = 10;
  m_lut_RI(98) = 10;
  m_lut_RI(137) = 10;
  m_lut_RI(152) = 10;
  m_lut_RI(196) = 10;
  // binary pattern with pattern 10101 in 8 bits and the rest 0s;
  m_lut_RI(21) = 11;
  m_lut_RI(42) = 11;
  m_lut_RI(69) = 11;
  m_lut_RI(81) = 11;
  m_lut_RI(84) = 11;
  m_lut_RI(138) = 11;
  m_lut_RI(162) = 11;
  m_lut_RI(168) = 11;
  // binary pattern with pattern 10111 in 8 bits and the rest 0s;
  m_lut_RI(23) = 12;
  m_lut_RI(46) = 12;
  m_lut_RI(92) = 12;
  m_lut_RI(113) = 12;
  m_lut_RI(139) = 12;
  m_lut_RI(184) = 12;
  m_lut_RI(197) = 12;
  m_lut_RI(226) = 12;
  // binary pattern with pattern 11001 in 8 bits and the rest 0s;
  m_lut_RI(25) = 13;
  m_lut_RI(35) = 13;
  m_lut_RI(50) = 13;
  m_lut_RI(70) = 13;
  m_lut_RI(100) = 13;
  m_lut_RI(140) = 13;
  m_lut_RI(145) = 13;
  m_lut_RI(200) = 13;
  // binary pattern with pattern 11011 in 8 bits and the rest 0s;
  m_lut_RI(27) = 14;
  m_lut_RI(54) = 14;
  m_lut_RI(99) = 14;
  m_lut_RI(108) = 14;
  m_lut_RI(141) = 14;
  m_lut_RI(177) = 14;
  m_lut_RI(198) = 14;
  m_lut_RI(216) = 14;
  // binary pattern with pattern 11101 in 8 bits and the rest 0s;
  m_lut_RI(29) = 15;
  m_lut_RI(58) = 15;
  m_lut_RI(71) = 15;
  m_lut_RI(116) = 15;
  m_lut_RI(142) = 15;
  m_lut_RI(163) = 15;
  m_lut_RI(209) = 15;
  m_lut_RI(232) = 15;
  // binary pattern with pattern 11111 in 8 bits and the rest 0s;
  m_lut_RI(31) = 16;
  m_lut_RI(62) = 16;
  m_lut_RI(124) = 16;
  m_lut_RI(143) = 16;
  m_lut_RI(199) = 16;
  m_lut_RI(227) = 16;
  m_lut_RI(241) = 16;
  m_lut_RI(248) = 16;
  // binary pattern with pattern 100101 in 8 bits and the rest 0s;
  m_lut_RI(37) = 17;
  m_lut_RI(41) = 17;
  m_lut_RI(73) = 17;
  m_lut_RI(74) = 17;
  m_lut_RI(82) = 17;
  m_lut_RI(146) = 17;
  m_lut_RI(148) = 17;
  m_lut_RI(164) = 17;
  // binary pattern with pattern 100111 in 8 bits and the rest 0s;
  m_lut_RI(39) = 18;
  m_lut_RI(57) = 18;
  m_lut_RI(78) = 18;
  m_lut_RI(114) = 18;
  m_lut_RI(147) = 18;
  m_lut_RI(156) = 18;
  m_lut_RI(201) = 18;
  m_lut_RI(228) = 18;
  // binary pattern with pattern 101011 in 8 bits and the rest 0s;
  m_lut_RI(43) = 19;
  m_lut_RI(86) = 19;
  m_lut_RI(89) = 19;
  m_lut_RI(101) = 19;
  m_lut_RI(149) = 19;
  m_lut_RI(172) = 19;
  m_lut_RI(178) = 19;
  m_lut_RI(202) = 19;
  // binary pattern with pattern 101101 in 8 bits and the rest 0s;
  m_lut_RI(45) = 20;
  m_lut_RI(75) = 20;
  m_lut_RI(90) = 20;
  m_lut_RI(105) = 20;
  m_lut_RI(150) = 20;
  m_lut_RI(165) = 20;
  m_lut_RI(180) = 20;
  m_lut_RI(210) = 20;
  // binary pattern with pattern 101111 in 8 bits and the rest 0s;
  m_lut_RI(47) = 21;
  m_lut_RI(94) = 21;
  m_lut_RI(121) = 21;
  m_lut_RI(151) = 21;
  m_lut_RI(188) = 21;
  m_lut_RI(203) = 21;
  m_lut_RI(229) = 21;
  m_lut_RI(242) = 21;
  // binary pattern with pattern 110011 in 8 bits and the rest 0s;
  m_lut_RI(51) = 22;
  m_lut_RI(102) = 22;
  m_lut_RI(153) = 22;
  m_lut_RI(204) = 22;
  // binary pattern with pattern 110101 in 8 bits and the rest 0s;
  m_lut_RI(53) = 23;
  m_lut_RI(77) = 23;
  m_lut_RI(83) = 23;
  m_lut_RI(106) = 23;
  m_lut_RI(154) = 23;
  m_lut_RI(166) = 23;
  m_lut_RI(169) = 23;
  m_lut_RI(212) = 23;
  // binary pattern with pattern 110111 in 8 bits and the rest 0s;
  m_lut_RI(55) = 24;
  m_lut_RI(110) = 24;
  m_lut_RI(115) = 24;
  m_lut_RI(155) = 24;
  m_lut_RI(185) = 24;
  m_lut_RI(205) = 24;
  m_lut_RI(220) = 24;
  m_lut_RI(230) = 24;
  // binary pattern with pattern 111011 in 8 bits and the rest 0s;
  m_lut_RI(59) = 25;
  m_lut_RI(103) = 25;
  m_lut_RI(118) = 25;
  m_lut_RI(157) = 25;
  m_lut_RI(179) = 25;
  m_lut_RI(206) = 25;
  m_lut_RI(217) = 25;
  m_lut_RI(236) = 25;
  // binary pattern with pattern 111101 in 8 bits and the rest 0s;
  m_lut_RI(61) = 26;
  m_lut_RI(79) = 26;
  m_lut_RI(122) = 26;
  m_lut_RI(158) = 26;
  m_lut_RI(167) = 26;
  m_lut_RI(211) = 26;
  m_lut_RI(233) = 26;
  m_lut_RI(244) = 26;
  // binary pattern with pattern 111111 in 8 bits and the rest 0s;
  m_lut_RI(63) = 27;
  m_lut_RI(126) = 27;
  m_lut_RI(159) = 27;
  m_lut_RI(207) = 27;
  m_lut_RI(231) = 27;
  m_lut_RI(243) = 27;
  m_lut_RI(249) = 27;
  m_lut_RI(252) = 27;
  // binary pattern with pattern 1010101 in 8 bits and the rest 0s;
  m_lut_RI(85) = 28;
  m_lut_RI(170) = 28;
  // binary pattern with pattern 1010111 in 8 bits and the rest 0s;
  m_lut_RI(87) = 29;
  m_lut_RI(93) = 29;
  m_lut_RI(117) = 29;
  m_lut_RI(171) = 29;
  m_lut_RI(174) = 29;
  m_lut_RI(186) = 29;
  m_lut_RI(213) = 29;
  m_lut_RI(234) = 29;
  // binary pattern with pattern 1011011 in 8 bits and the rest 0s;
  m_lut_RI(91) = 30;
  m_lut_RI(107) = 30;
  m_lut_RI(109) = 30;
  m_lut_RI(173) = 30;
  m_lut_RI(181) = 30;
  m_lut_RI(182) = 30;
  m_lut_RI(214) = 30;
  m_lut_RI(218) = 30;
  // binary pattern with pattern 1011111 in 8 bits and the rest 0s;
  m_lut_RI(95) = 31;
  m_lut_RI(125) = 31;
  m_lut_RI(175) = 31;
  m_lut_RI(190) = 31;
  m_lut_RI(215) = 31;
  m_lut_RI(235) = 31;
  m_lut_RI(245) = 31;
  m_lut_RI(250) = 31;
  // binary pattern with pattern 1101111 in 8 bits and the rest 0s;
  m_lut_RI(111) = 32;
  m_lut_RI(123) = 32;
  m_lut_RI(183) = 32;
  m_lut_RI(189) = 32;
  m_lut_RI(219) = 32;
  m_lut_RI(222) = 32;
  m_lut_RI(237) = 32;
  m_lut_RI(246) = 32;
  // binary pattern with pattern 1110111 in 8 bits and the rest 0s;
  m_lut_RI(119) = 33;
  m_lut_RI(187) = 33;
  m_lut_RI(221) = 33;
  m_lut_RI(238) = 33;
  // binary pattern with pattern 1111111 in 8 bits and the rest 0s;
  m_lut_RI(127) = 34;
  m_lut_RI(191) = 34;
  m_lut_RI(223) = 34;
  m_lut_RI(239) = 34;
  m_lut_RI(247) = 34;
  m_lut_RI(251) = 34;
  m_lut_RI(253) = 34;
  m_lut_RI(254) = 34;
  // binary pattern with pattern 11111111 in 8 bits
  m_lut_RI(255) = 35;
}


void bob::ip::LBP8R::init_lut_U2()
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

void bob::ip::LBP8R::init_lut_U2RI()
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

void bob::ip::LBP8R::init_lut_add_average_bit()
{
  m_lut_add_average_bit.resize(512);
  blitz::firstIndex i;
  m_lut_add_average_bit = i;
}

void bob::ip::LBP8R::init_lut_normal()
{
  m_lut_normal.resize(256);
  blitz::firstIndex i;
  m_lut_normal = i;
}
