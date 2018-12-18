# http://www.sfu.ca/~ssurjano/marthe.html
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %
# % MARTHE DATASET READ-IN
# %
# % Authors: Sonja Surjanovic, Simon Fraser University
# %          Derek Bingham, Simon Fraser University
# % Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
# %
# % Copyright 2013. Derek Bingham, Simon Fraser University.
# %
# % THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
# % FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
# % derivative works, such modified software should be clearly marked.
# % Additionally, this program is free software; you can redistribute it
# % and/or modify it under the terms of the GNU General Public License as
# % published by the Free Software Foundation; version 2.0 of the License.
# % Accordingly, this program is distributed in the hope that it will be
# % useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# % of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# % General Public License for more details.
# %
# % For function details and reference information, see:
# % http://www.sfu.ca/~ssurjano/
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %
# % OBSERVATIONS: 300
# %
# % INPUT VARIABLES:
# %
# % per1  = hydraulic conductivity layer 1
# % per2  = hydraulic conductivity layer 2
# % per3  = hydraulic conductivity layer 3
# % perz1 = hydraulic conductivity zone 1
# % perz2 = hydraulic conductivity zone 2
# % perz3 = hydraulic conductivity zone 3
# % perz4 = hydraulic conductivity zone 4
# % d1    = longitudinal dispersivity layer 1
# % d2    = longitudinal dispersivity layer 2
# % d3    = longitudinal dispersivity layer 3
# % dt1   = transversal dispersivity layer 1
# % dt2   = transversal dispersivity layer 2
# % dt3   = transversal dispersivity layer 3
# % kd1   = volumetric distribution coefficient 1.1
# % kd2   = volumetric distribution coefficient 1.2
# % kd3   = volumetric distribution coefficient 1.3
# % poros = porosity
# % i1    = infiltration type 1
# % i2    = infiltration type 2
# % i3    = infiltration type 3
# %
# % OUTPUT VARIABLES:
# %
# % p102K
# % p104
# % p106
# % p2_76
# % p29K
# % p31K
# % p35K
# % p37K
# % p38
# % p4b
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# marthe = importdata('marthedata.txt');
# marthedata = marthe.data;
#
# per1  = marthedata(:, 1);
# per2  = marthedata(:, 2);
# per3  = marthedata(:, 3);
# perz1 = marthedata(:, 4);
# perz2 = marthedata(:, 5);
# perz3 = marthedata(:, 6);
# perz4 = marthedata(:, 7);
# d1    = marthedata(:, 8);
# d2    = marthedata(:, 9);
# d3    = marthedata(:, 10);
# dt1   = marthedata(:, 11);
# dt2   = marthedata(:, 12);
# dt3   = marthedata(:, 13);
# kd1   = marthedata(:, 14);
# kd2   = marthedata(:, 15);
# kd3   = marthedata(:, 16);
# poros = marthedata(:, 17);
# i1    = marthedata(:, 18);
# i2    = marthedata(:, 19);
# i3    = marthedata(:, 20);
#
# p102K = marthedata(:, 21);
# p104  = marthedata(:, 22);
# p106  = marthedata(:, 23);
# p2_76 = marthedata(:, 24);
# p29K  = marthedata(:, 25);
# p31K  = marthedata(:, 26);
# p35K  = marthedata(:, 27);
p37K  = marthedata(:, 28);
p38   = marthedata(:, 29);
p4b   = marthedata(:, 30);"