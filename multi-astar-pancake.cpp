#include <bits/stdc++.h>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <stdbool.h>
#include <string>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <vector>

using namespace std;
typedef long long ll;
typedef vector<ll> vll;

#define NMAX 15

int SORTED_STACK[NMAX] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
ll SORTED_STACK_INT[NMAX];
int UPPER_BOUNDS[NMAX] = {0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18};

typedef struct {
  unsigned char p[NMAX], dep, lb;
} pancake_t;

int get_lb(int target_len, int current_len);
void display_vll(vll v, int len);
void manage_stacks(int target_len, vll current_stacks, int start_len);
vll init_stacks(int len, int lb);
vll manage_layer(vll stacks, int nstacks, int lb, int len,
                        int rank, int size);
int distance(int len, int *p, int ub, int reqmin);
void inplace_flip(int *p, int i);
void flip(int *p, int *np, int len, int i);
vll handle_root(int target_len, int start_len);
void handle_worker(int rank);

// Return current time, for performance measurement
uint64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
}

const MPI_Comm comm = MPI_COMM_WORLD;
const int root = 0;

int main(int argc, char **argv) {
  int target_len, start_len, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(comm, &rank);

  if (rank == root) {
    uint64_t start = GetTimeStamp();
    cin >> target_len;
    cin >> start_len;
    vll res = handle_root(target_len, start_len);
    uint64_t time_taken = (uint64_t) (GetTimeStamp() - start);
    display_vll(res, target_len);
		printf("Time: %ld us\n", time_taken);
  } else {
    handle_worker(rank);
  }

  MPI_Finalize();
  return 0;
}

vll handle_root(int target_len, int start_len) {
  int rank, size, lb = get_lb(target_len, start_len + 1), len = start_len;

  // handle MPI Comm
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // init stack
  vll current_stack = init_stacks(len, lb);
  int stack_size = current_stack.size();

  // init control var
  int recv_counts[size];
  int disps[size], i;

  // BCast start and target lengths
  int start[2] = {len, target_len};
  MPI_Bcast(start, 2, MPI_INT, root, comm);

  // get working
  while (len < target_len) {

    // broadcast init information
    int init[4] = {start_len, target_len, stack_size, lb};
    MPI_Bcast(init, 4, MPI_INT, root, comm);

    // BCast current stack (containing every element)
    MPI_Bcast(&current_stack[0], stack_size, MPI_LONG_LONG_INT, root, comm);


    // Generate the next stack (partial)
    vll next_stack =
        manage_layer(current_stack, stack_size, lb, len, rank, size);

    stack_size = next_stack.size();


    // gather all generated stack sizes from processes
    MPI_Gather(&stack_size, 1, MPI_INT, recv_counts, 1, MPI_INT, root, comm);

    // calculate displacements
    disps[0] = 0;
    for (i = 1; i < size; ++i) {
      disps[i] = disps[i - 1] + recv_counts[i - 1];
    }

    stack_size = 0;
    for (i = 0; i < size; ++i)
      stack_size += recv_counts[i];

    ll receive[stack_size];

    // gather all current stacks into this layer
    MPI_Gatherv(&next_stack[0], recv_counts[0], MPI_LONG_LONG_INT, receive,
                recv_counts, disps, MPI_LONG_LONG_INT, root, comm);

    vll last_stack(receive, receive + stack_size);
    current_stack = last_stack;

    ++len;
    lb = get_lb(target_len, len + 1);
  }
  return current_stack;
}

void handle_worker(int rank) {
  int size;
  MPI_Comm_size(comm, &size);
  int init[4], target_len, stack_size, len, lb;
  vll current_stack;

  // BCast start and target lengths
  int start[2];
  MPI_Bcast(start, 2, MPI_INT, root, comm);
  len = start[0], target_len = start[1];

  while (len < target_len) {
    MPI_Bcast(init, 4, MPI_INT, root, comm);
    stack_size = init[2], lb = init[3];
    ll receive[stack_size];


    // BCast current stack (containing every element)
    MPI_Bcast(receive, stack_size, MPI_LONG_LONG_INT, root, comm);

    ll total = 0;
    for (int i = 0; i < stack_size; ++i)
      total += receive[i];

    vll current_stack(receive, receive + stack_size);
    // Generate the next stack(partial)
    vll next_stack =
        manage_layer(current_stack, stack_size, lb, len, rank, size);

    stack_size = next_stack.size();


    // gather all generated stack sizes from processes
    MPI_Gather(&stack_size, 1, MPI_INT, NULL, 1, MPI_INT, root, comm);

    // gather all current stacks into this layer
    MPI_Gatherv(&next_stack[0], stack_size, MPI_LONG_LONG_INT, NULL, NULL, NULL,
                NULL, root, comm);
    ++len;
  }
}

/* =========================[ HELPER FUNCS ]==================================
 */

void inplace_flip(int *p, int index) {
  int i, tmp;

  for (i = 0; i < ((index + 1) >> 1); i++) {
    tmp = p[index - i];
    p[index - i] = p[i];
    p[i] = tmp;
  }
}

void flip(int *p, int *np, int len, int index) {
  int i;

  for (i = 0; i <= index; i++)
    np[i] = p[index - i];
  for (; i < len; i++)
    np[i] = p[i];
}

int get_lb(int target_len, int current_len) {
  int current_bound = UPPER_BOUNDS[target_len - 1];
  while (target_len > current_len) {
    current_bound -= 2;
    --target_len;
  }
  return max(current_bound, 0);
}

/* returns number of the stack in lex. ordering */
ll perm_to_int(int len, int *p) {
  int i, j;
  ll ret, tmp;
  int used[NMAX + 2];

  for (i = 0; i < len; i++)
    used[i] = 0;
  ret = 0;
  for (i = 0; i < len; i++) {
    tmp = 0;
    for (j = 0; j < p[i]; j++)
      if (!used[j])
        tmp++;
    ret = ret * (len - i) + tmp;
    used[p[i]] = 1;
  }

  return ret;
}

void int_to_perm(int len, long long start, int *p) {
  int i, j, tmp;
  ll a = start;
  int used[NMAX + 2], val[NMAX + 2];

  for (i = 0; i < len; i++)
    used[i] = 0;
  for (i = len - 1; i >= 0; i--) {
    val[i] = a % (len - i);
    a /= (len - i);
  }
  for (i = 0; i < len; i++) {
    tmp = val[i];
    for (j = 0; j < len && tmp > 0; j++)
      if (!used[j])
        tmp--;
    while (used[j])
      j++;
    p[i] = j;
    used[j] = 1;
  }
}
void show_perm_int(ll perm, int len) {
  int tmp[len + 2];
  int_to_perm(len, perm, tmp);
  for (int i = 0; i < len; ++i)
    cout << tmp[i] << ", ";
  cout << endl;
}

ll flip_int(int len, ll lex, int index) {
  int tmp[len];
  int_to_perm(len, lex, tmp);
  inplace_flip(tmp, index);
  ll res = perm_to_int(len, tmp);
  return res;
}

void display_vll(vll v, int len) {
  for (int i = 0; i < (int)v.size(); ++i)
    show_perm_int(v[i], len);
}

// Adapated from Joseph Cibulka - On average and highest number of flips in pancake sorting
int get_upper_bound(int len, int *p, int remaining_adj, int wastes, int ubbt_dep,
                   int ubbt_end_dep) {
  int np[NMAX + 2];
  int i, res, tmp, found = 0;
  int joined = 0;
  res = 0;

  // try adding adjacency
  for (i = 2; i < len; i++)
    if (abs(p[i] - p[0]) == 1) {
      found++;
      if (abs(p[i] - p[i - 1]) != 1) {
        flip(p, np, len, i - 1);
        joined = 1;
        tmp = 1 + get_upper_bound(len, np, remaining_adj - 1, wastes, ubbt_dep,
                                 ubbt_end_dep);
        res = res > tmp ? res : tmp;
        if (res == remaining_adj)
          return res;
      }
      if (found == 2)
        break;
    }
  if (p[0] == len - 1 && len > 1) {
    flip(p, np, len, len - 1);
    joined = 1;
    tmp = 1 + get_upper_bound(len - 1, np, remaining_adj - 1, wastes, ubbt_dep,
                             ubbt_end_dep);
    res = res > tmp ? res : tmp;
  }
  // try a waste
  if (wastes > 0 && (remaining_adj <= ubbt_dep ||
                     (remaining_adj <= ubbt_end_dep && !joined))) {
    for (i = 2; i < len; i++)
      if (abs(p[i] - p[0]) != 1 && abs(p[i] - p[i - 1]) != 1) {
        flip(p, np, len, i - 1);
        tmp = get_upper_bound(len, np, remaining_adj, wastes - 1, ubbt_dep,
                             ubbt_end_dep);
        res = res > tmp ? res : tmp;
        if (res == remaining_adj)
          return res;
      }
    if (p[0] != len - 1 && p[len - 1] != len - 1) {
      flip(p, np, len, len - 1);
      tmp = get_upper_bound(len, np, remaining_adj, wastes - 1, ubbt_dep,
                           ubbt_end_dep);
      res = res > tmp ? res : tmp;
    }
  }
  return res;
}

class compare {
public:
  int operator()(pancake_t el1, pancake_t el2) {
    int tmp;
    tmp = el1.lb - el2.lb;
    if (tmp < 0)
      return 1;
    if (tmp > 0)
      return -1;
    tmp = el2.dep - el1.dep;
    if (tmp < 0)
      return 1;
    if (tmp > 0)
      return -1;
    return 0;
  }
};

int heuristic_ub(int len, int *p, int remaining_adj) {
  int np[NMAX + 2];
  int i, res, tmp, found = 0;

  res = 0;
  for (i = 2; i < len; i++)
    if (abs(p[i] - p[0]) == 1) {
      found++;
      if (abs(p[i] - p[i - 1]) != 1) {
        flip(p, np, len, i - 1);
        tmp = 1 + heuristic_ub(len, np, remaining_adj - 1);
        res = res > tmp ? res : tmp;
        if (res == remaining_adj)
          return res;
      }
      if (found == 2)
        break;
    }
  if (p[0] == len - 1 && len > 1) {
    flip(p, np, len, len - 1);
    tmp = 1 + heuristic_ub(len - 1, np, remaining_adj - 1);
    res = res > tmp ? res : tmp;
  }
  return res;
}

int is_adj(int len, int *p, int a) {
  if (a == len - 1) {
    if (p[a] == len - 1)
      return 1;
    else
      return 0;
  } else {
    if (abs(p[a] - p[a + 1]) == 1)
      return 1;
    return 0;
  }
}

int count_adj(int len, int *p) {
  int i, ret = 0;
  for (i = 0; i < len; i++)
    ret += is_adj(len, p, i);
  return ret;
}

ll append_len(ll lex, int len) {
  int tmp[len + 1];

  int_to_perm(len, lex, tmp);
  tmp[len] = len - 1;

  return perm_to_int(len + 1, tmp);
}
/* =========================[ IMPURE FUNCS ]==================================
 */

void push_heap_lb(
    int len, int *p, int dep, int known_lb, int asreqmin, int *asres,
    priority_queue<pancake_t, vector<pancake_t>, compare> &q) {

  int i, remaining_adj;
  int tmplow, tmpup;
  pancake_t *newelem;

  tmpup = -1;

  if (known_lb >= 0) {
    tmplow = known_lb;
    if (tmplow >= *asres)
      return; 
  } else {
    remaining_adj = len - count_adj(len, p);
    tmplow = remaining_adj + dep;
    if (tmplow >= *asres)
      return;

    if (heuristic_ub(len, p, remaining_adj) == remaining_adj)
      tmpup = tmplow;
    else {
      tmplow++;
      if (tmplow >= *asres)
        return;
    }

    if (tmpup == -1) {
      if (get_upper_bound(len, p, remaining_adj, 1, 4, 6) == remaining_adj)
        tmpup = remaining_adj + dep + 1;
    }

    if (tmplow >= *asres)
      return;
    if (tmpup >= 0) {
      if (tmpup < *asres) { 
        *asres = tmpup;
        if (*asres < asreqmin)
          return;
      }
      if (tmpup == tmplow) {
        return;
      }
    }
  }
  // add to heap
  newelem = new pancake_t;
  for (i = 0; i < len; i++)
    newelem->p[i] = (char)p[i];
  newelem->dep = (char)dep;
  newelem->lb = (char)tmplow;
  q.push(*newelem);
}
void asearch(int len, int asreqmin, int *asres,
             priority_queue<pancake_t, vector<pancake_t>, compare> &q) {
  int i, tmp;
  int p[NMAX + 2];
  int np[NMAX + 2];
  pancake_t el;

  while (q.size() > 0) {
    el = q.top();
    q.pop();
    if (el.lb >= *asres) {
      return;
    }
    for (i = 0; i < len; i++)
      p[i] = el.p[i];
    for (i = 0; i < len; i++)
      if (p[i] != i)
        break;
    if (i == len) {
      assert(el.dep < *asres);
      *asres = el.dep;
      return;
    }

    for (i = 1; i < len; i++) // flip pancakes 0...i
    {
      flip(p, np, len, i);
      tmp = -1;
      if (is_adj(len, np, i) && !is_adj(len, p, i))
        tmp = el.lb;
      push_heap_lb(len, np, el.dep + 1, tmp, asreqmin, asres, q);
      if (*asres < asreqmin) {
        return;
      }
    }
  }
}

// start the A*search
int distance(int len, int *p, int ub, int reqmin) {

  int asres = ub;
  int asreqmin = reqmin;

  priority_queue<pancake_t, vector<pancake_t>, compare> q;
  push_heap_lb(len, p, 0, -1, asreqmin, &asres, q);

  asearch(len, asreqmin, &asres, q);

  return asres;
}

// BFS and return vll where distance from start >= lb
vll init_stacks(int len, int lb) {
  ll start = SORTED_STACK_INT[len - 1], nxt, cur;
  int cur_dist, max_dist = 0;
  vll res;
  unordered_map<ll, bool> visited;
  unordered_map<ll, int> D;
  queue<ll> q;

  q.push(start);
  D[start] = 0;

  while (q.size() > 0) {
    cur = q.front();
    q.pop();

    if (!visited[cur]) {
      visited[cur] = true;
      cur_dist = D[cur];
      max_dist = max(cur_dist, max_dist);

      // if its > lb then we want to return its value
      if (cur_dist >= lb)
        res.push_back(cur);

      for (int i = 1; i < len; ++i) {
        nxt = flip_int(len, cur, i);

        // if we haven't visited the generated stack then we want to
        if (!visited[nxt] && D.find(nxt) == D.end()) {
          D[nxt] = cur_dist + 1;
          q.push(nxt);
        }
      }
    }
  }
  return res;
}

vll manage_layer(vll stacks, int nstacks, int lb, int len,
                        int rank, int size) {
  int tmp_dist, max_dist = 0, p[NMAX + 2];
  ll cur, tmp, added = 0;
  unordered_map<ll, bool> seen;
  vll next_layer;

  for (int i = 0; i < nstacks; ++i) {
    if (i % size != rank)
      continue;

    cur = append_len(stacks[i], len);

    if (!seen[cur]) {
      seen[cur] = true;
      // get dist and update max_dist
      int_to_perm(len, cur, p);
      tmp_dist = distance(len, p, UPPER_BOUNDS[len - 1], lb);
      max_dist = tmp_dist > max_dist ? tmp_dist : max_dist;
      // check if it meets the LB
      if (tmp_dist >= lb)
        next_layer.push_back(cur), added++;
    }

    cur = flip_int(len + 1, cur, len);

    if (!seen[cur]) {
      seen[cur] = true;
      // flip the whole stack
      // get dist and update max_dist
      int_to_perm(len + 1, cur, p);
      tmp_dist = distance(len + 1, p, UPPER_BOUNDS[len], lb);
      max_dist = tmp_dist > max_dist ? tmp_dist : max_dist;
      // check if it meets the lb
      if (tmp_dist >= lb)
        next_layer.push_back(cur), added++;
    }

    for (int i = 1; i <= len; ++i) {
      tmp = flip_int(len + 1, cur, i);

      if (!seen[tmp]) {
        seen[tmp] = true;
        // get dist and update max_dist
        int_to_perm(len + 1, tmp, p);
        tmp_dist = distance(len + 1, p, UPPER_BOUNDS[len], lb);
        max_dist = tmp_dist > max_dist ? tmp_dist : max_dist;

        // check if it meets the lb + 1
        if (tmp_dist >= lb)
          next_layer.push_back(tmp), added++;
      }
    }
  }

  return next_layer;
}
