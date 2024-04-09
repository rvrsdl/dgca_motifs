% Demonstrating the DGCAv2 model using Kronecker tensor products.

% Setup
s = 2; # number of states
nsteps = 50;
max_size = 1024;

# Define quadrant matrices
Qm = [1,0;0,0];
Qf = [0,1;0,0];
Qb = [0,0;1,0];
Qn = [0,0;0,1];

% Initialise SLPs
action_slp = rand(15, 3*s+1)*2-1;
state_slp = rand(s, 3*s+1)*2-1;

% Set seed graph adjacency matrix and state matrix
A = [0,1,0;
     0,0,1;
     1,0,0];
S = [1,1,0;
     0,0,1];

for step=1:nsteps     
  # Begin update cycle
  % 1. Gather neighbourhood information    
  C = vertcat(S, S*A, S*A', ones(1,columns(S)));
  % 2. Run action SLP
  action_output = action_slp * C;
  % 2a. Interpret output
  K = vertcat(
    action_output(1:3,:) == max(action_output(1:3,:),[],1),
    action_output(4:7,:) == max(action_output(4:7,:),[],1),
    action_output(8:11,:) == max(action_output(8:11,:),[],1),
    action_output(12:15,:) == max(action_output(12:15,:),[],1));
  % - action choices
  remove = K(1,:);
  stay   = K(2,:);
  divide = K(3,:);
  keep = [not(remove), divide];
  if sum(keep)>max_size
    break;
  endif
  % - new node wiring
  k_fi = K(5, :);
  k_fa = K(6, :);
  k_ft = K(7, :);
  k_bi = K(9, :);
  k_ba = K(10,:);
  k_bt = K(11,:);
  k_ni = K(13,:);
  k_na = K(14,:);
  k_nt = K(15,:);
  % Restructure matrix
  I = eye(rows(A));
  A = kron(Qm, A) ...
    + kron(Qf, (I*diag(k_fi) + A*diag(k_fa) + A'*diag(k_ft))) ...
    + kron(Qb, (I*diag(k_bi) + A*diag(k_ba) + A'*diag(k_bt))) ...
    + kron(Qn, (I*diag(k_ni) + A*diag(k_na) + A'*diag(k_nt)));  
  A = A(keep, keep);
  S = [S,S];
  S = S(:,keep);
  % 3. Gather neighbourhood information
  C = vertcat(S, S*A, S*A', ones(1,columns(S)));
  % 4. Run state SLP
  state_output = state_slp * C;
  S = state_output==max(state_output,[],1);
  % Update cycle complete
endfor  