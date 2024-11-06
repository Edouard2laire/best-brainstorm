function [D, dD,H] = be_free_energy2(lambda, M, noise_var,G_active_var_Gt, clusters,  varargin)
%CALCULATE_FREE_ENERGY gives the free energy of a system
%   [D, dD] = CALCULATE_FREE_ENERGY(LAMBDA, M, NOISE_VAR, CLUSTERS,
%   NB_CLUSTERS)
%   calculates the free energy of the system for a given LAMBDA and 
%   returns it in D. The function also returns the derivative of D with 
%   respect to LAMBDA in dD.  This method should not be called by itself.
%
%   The method uses the methodoly described in :
%   Amblard, C., Lapalme, E., Lina, J. 2004. Biomagnetic Source Detection
%       by Maximyum Entropy and Graphical  Models, IEEE Transactions on 
%       Biomedical Engineering, vol. 51, no 3, p. 427-442.
%
%   The formulas are :
%       D(lambda) = lambda' * M - 
%                   (1/2)* noise_var * lambda' * lambda - 
%                   sum(F* * (G' * lambda))
%
%       F*(xi) = ln[(1- alpha) * exp(F0) + alpha * exp(F1)]
%           where F0 is the inactive state and
%                 F1 is the active state
%       F0(xi) = 1/2 * xi' * omega * xi
%       F1(xi) = 1/2 * xi' * sigma * xi + xi' * mu
%
%% ==============================================
% Copyright (C) 2011 - LATIS Team
%
%  Authors: LATIS team, 2011
%
%% ==============================================
% License 
%
% BEst is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    BEst is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with BEst. If not, see <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------

lambda_trans = lambda';      
isUsingActiveMean = ~isempty(clusters(1).active_mean);
isUsingInactiveVar = ~isempty(clusters(1).inactive_var);


% Estimate dF1 and F1 (separating the contribution  of the mean and
% covariance for optimization purpose)
dF1     = squeeze(pagemtimes(G_active_var_Gt,lambda));
F1    =  1/2 * lambda_trans*dF1as; 

if isUsingActiveMean

    dF1b = zeros(size(dF1as));
    for ii = 1:size(dF1b,2)

        active_mean = clusters(ii).active_mean;
        dF1b(:,ii)  = clusters(ii).G * active_mean;

    end
    dF1 = dF1 + dF1b;
    F1  = F1 + lambda_trans * dF1b;

end

% Estimate F0
% F0 is set to a dirac by default (omega=0).
    if isempty(omega)
        F0=0;
    else
        F0 = 1/2 * xi' * omega * xi;
    end

p                   = [clusters.active_probability];
coeffs_free_energy  = (1-p) .* exp(-F1)  +  p;

F1 = sum(F1 + log(coeffs_free_energy));

s   = p ./ coeffs_free_energy;
dF  = s .* dF1;


dD  =                M - sum(dF,2) - noise_var * lambda;
D   = lambda_trans * M - sum(F1) - (1/2) * lambda_trans * noise_var * lambda;

% The outcome of the equations produces a strictly convex function
% (with a maximum).
D   = -D;
dD  = -dD;

if nargout >=3
    disp('using hessian')
    H = sum(G_active_var_Gt,3) + noise_var;
end


end