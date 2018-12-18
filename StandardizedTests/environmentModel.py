# http://www.sfu.ca/~ssurjano/environ.html

# function [y] = environ(xx, s, t)
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %
# % ENVIRONMENTAL MODEL FUNCTION
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
#     %
# % For function details and reference information, see:
# % http://www.sfu.ca/~ssurjano/
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %
# % OUTPUT AND INPUTS:
# %
# % y = row vector of scaled concentrations of the pollutant at the
# %     space-time vectors (s, t)
# %     Its structure is:
# %     y(s_1, t_1), y(s_1, t_2), ..., y(s_1, t_dt), y(s_2, t_1), ...,
# %     y(s_2,t_dt), ..., y(s_ds, t_1), ..., y(s_ds, t_dt)
# % xx = [M, D, L, tau]
# % s = vector of locations (optional), with default value
# %     [0.5, 1, 1.5, 2, 2.5]
# % t = vector of times (optional), with default value
# %     [0.3, 0.6, ..., 50.7, 60]
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# M   = xx(1);
# D   = xx(2);
# L   = xx(3);
# tau = xx(4);
#
# if (nargin < 3)
#     t = [0.3:0.3:60];
#     end
#     if (nargin < 2)
#         s = [0.5, 1, 1.5, 2, 2.5];
#     end
#
#     ds = length(s);
#     dt = length(t);
#     dY = ds * dt;
#     Y = zeros(ds, dt);
#
#     % Create matrix Y, where each row corresponds to si and each column
#     % corresponds to tj.
#     for (ii = 1:ds)
#     si = s(ii);
#     for (jj = 1:dt)
#     tj = t(jj);
#
#     term1a = M / sqrt(4*pi*D*tj);
#     term1b = exp(-si^2 / (4*D*tj));
#     term1 = term1a * term1b;
#
#     term2 = 0;
#     if (tau < tj)
#         term2a = M / sqrt(4*pi*D*(tj-tau));
#         term2b = exp(-(si-L)^2 / (4*D*(tj-tau)));
#         term2 = term2a * term2b;
#     end
#
    C = term1 + term2;
    Y(ii, jj) = sqrt(4*pi) * C;
end
end

% Convert the matrix into a vector (by rows).
Yrow = Y';
y = Yrow(:)';

end
