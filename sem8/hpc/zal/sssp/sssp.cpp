#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>
#include <numeric>
#include <set>
#include <unordered_map>
#include <vector>

using namespace std;

// aliases {{{
using ll = long long;
using pll = pair<ll, ll>;
using vpil = vector<pll>;
using vvpil = vector<vpil>;
// }}}
// params {{{
static ll Delta = DELTA;
static constexpr ll INF = 2e18;
// }}}
// global variables {{{
static int my_rank, world_size; // MPI rank and total processes
static ll k = 0;                // current bucket index
static map<ll, set<ll>> B;      // bucket index -> {vertices of such index}
static ll v_first, v_last;      // first and last vertices owned by this process
static ll v_cnt;                // number of vertices owned by this process
static ll n;                    // number of vertices in graph
static vvpil adj_local_short;   // local short edges
static vvpil adj_local_long;    // local long edges
static vvpil adj_remote_short;  // remote short edges
static vvpil adj_remote_long;   // remote long edges
static vector<ll> d;            // tentative distances of owned vertices
static set<ll> updated;         // vertices changed in the previous step
static vector<ll> bounds;       // vertex bounds of each rank

static MPI_Datatype MPI_UPDATE_TYPE;
static MPI_Datatype MPI_REQUEST_TYPE;

static constexpr auto wcmp = [](const pll &a, const pll &b) {
    return a.second < b.second;
};

static struct {
    ll processed;
    ll threshold;
} hybridization = {0, 0};

struct Request {
    ll target;
    ll source;
    ll weight;
};
// }}}
// utility functions {{{
void compute_boundaries()
{
    vector<ll> all_firsts(world_size);
    MPI_Allgather(&v_first, 1, MPI_LONG_LONG, all_firsts.data(), 1,
                  MPI_LONG_LONG, MPI_COMM_WORLD);
    bounds.resize(world_size + 1);
    for (int i = 0; i < world_size; i++)
        bounds[i] = all_firsts[i];
    bounds[world_size] = n;
}

bool is_local(ll v) { return v_first <= v && v <= v_last; }

int get_owner(ll v)
{
    static unordered_map<ll, int> owner;
    if (!owner.count(v)) {
        auto it = upper_bound(bounds.begin(), bounds.end(), v);
        owner[v] = distance(bounds.begin(), it) - 1;
    }
    return owner[v];
}
// }}}
// io {{{
void load_graph(const string &input_file)
{
    ifstream in(input_file);
    if (!in.is_open()) {
        cerr << "Process " << my_rank << " failed to open file." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    in >> n >> v_first >> v_last;
    v_cnt = v_last - v_first + 1;
    adj_local_short.resize(v_cnt);
    adj_local_long.resize(v_cnt);
    adj_remote_short.resize(v_cnt);
    adj_remote_long.resize(v_cnt);

    ll u, v, w;
    set<tuple<ll, ll, ll>> edges;
    while (in >> u >> v >> w) {
        bool u_local = is_local(u);
        bool v_local = is_local(v);
        if (u_local)
            edges.emplace(u, v, w);
        if (v_local)
            edges.emplace(v, u, w);
    }
    for (const auto &[u, v, w] : edges) {
        bool v_local = is_local(v);
        auto &adj = w < Delta ? (v_local ? adj_local_short : adj_remote_short)
                              : (v_local ? adj_local_long : adj_remote_long);
        adj[u - v_first].emplace_back(v, w);
    }

    d.assign(v_cnt, INF);

    hybridization.processed = 0;
    hybridization.threshold = (ll)((double)n * HYBRIDIZATION_TAU);
}

void write_output(const string &output_file)
{
    ofstream out(output_file);
    for (int i = 0; i < v_cnt; i++)
        out << d[i] << "\n";
}
// }}}
void update_local(ll g_v, ll dist) // {{{
{
    int l_v = g_v - v_first;
    if (l_v < 0 || l_v >= v_cnt)
        return;

    if (dist < d[l_v]) {
        ll old_bucket = d[l_v] / Delta;
        d[l_v] = dist;
        ll new_bucket = dist / Delta;

        // Only move if bucket changes
        if (old_bucket != new_bucket) {
            B[old_bucket].erase(g_v);
            if (B[old_bucket].empty())
                B.erase(old_bucket);
            B[new_bucket].insert(g_v);
        }

        if (new_bucket == k)
            updated.insert(g_v);
    }
}
// }}}
template <bool SHORT, bool LOCAL> // {{{
void relax(const set<ll> &A, vector<vector<pll>> &send_updates)
{
    for (ll g_u : A) {
        int u = g_u - v_first;
        if (d[u] == INF)
            continue;

        if constexpr (SHORT) {
            const auto &adj = LOCAL ? adj_local_short : adj_remote_short;
            for (const auto &[v, w] : adj[u]) {
                ll new_dist = d[u] + w;

                if constexpr (ENABLE_IOS) {
                    if (new_dist >= (k + 1) * Delta)
                        break;
                }

                if constexpr (LOCAL)
                    update_local(v, new_dist);
                else
                    send_updates[get_owner(v)].emplace_back(v, new_dist);
            }
        } else if constexpr (ENABLE_IOS) {
            const auto &adj = LOCAL ? adj_local_short : adj_remote_short;
            for (auto it = adj[u].rbegin(); it != adj[u].rend(); ++it) {
                const auto &[v, w] = *it;
                ll new_dist = d[u] + w;

                if constexpr (ENABLE_IOS) {
                    if (new_dist < (k + 1) * Delta)
                        break;
                }

                if constexpr (LOCAL)
                    update_local(v, new_dist);
                else
                    send_updates[get_owner(v)].emplace_back(v, new_dist);
            }
        }

        if constexpr (!SHORT) {
            const auto &adj = LOCAL ? adj_local_long : adj_remote_long;
            for (const auto &[v, w] : adj[u]) {
                ll new_dist = d[u] + w;

                if constexpr (LOCAL)
                    update_local(v, new_dist);
                else
                    send_updates[get_owner(v)].emplace_back(v, new_dist);
            }
        }
    }
}
// }}}
void communicate_updates(vector<vector<pll>> &send_updates) // {{{
{
    vector<int> send_cnt(world_size);
    vector<int> recv_cnt(world_size);
    vector<int> send_displs(world_size + 1);
    vector<int> recv_displs(world_size + 1);
    vector<pll> send_buf;
    vector<pll> recv_buf;

    for (int i = 0; i < world_size; i++)
        send_cnt[i] = send_updates[i].size();

    MPI_Alltoall(send_cnt.data(), 1, MPI_INT, recv_cnt.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    send_displs[0] = 0;
    recv_displs[0] = 0;
    partial_sum(send_cnt.begin(), send_cnt.end(), send_displs.begin() + 1);
    partial_sum(recv_cnt.begin(), recv_cnt.end(), recv_displs.begin() + 1);

    send_buf.resize(send_displs[world_size]);
    recv_buf.resize(recv_displs[world_size]);

    for (int i = 0, idx = 0; i < world_size; i++) {
        for (const auto &u : send_updates[i])
            send_buf[idx++] = u;
    }

    MPI_Alltoallv(send_buf.data(), send_cnt.data(), send_displs.data(),
                  MPI_UPDATE_TYPE, recv_buf.data(), recv_cnt.data(),
                  recv_displs.data(), MPI_UPDATE_TYPE, MPI_COMM_WORLD);

    for (const auto &u : recv_buf)
        update_local(u.first, u.second);
}
// }}}
bool push_vs_pull() // {{{
{
    ll push_volume = 0, pull_volume = 0;
    const auto &adj_long = adj_remote_long;

    for (ll g_u : B[k]) {
        int l_u = g_u - v_first;
        if (d[l_u] == INF)
            continue;
        push_volume += adj_long[l_u].size();
    }

    for (auto it = B.upper_bound(k); it != B.end(); ++it) {
        for (ll g_u : it->second) {
            int l_u = g_u - v_first;
            ll threshold = d[l_u] - k * Delta;
            auto jt = lower_bound(adj_long[l_u].begin(), adj_long[l_u].end(),
                                  make_pair(0, threshold), wcmp);
            pull_volume += distance(adj_long[l_u].begin(), jt) * 2;

            if constexpr (!ENABLE_IOS)
                continue;

            const auto &adj_short = adj_remote_short;
            auto r = lower_bound(adj_short[l_u].begin(), adj_short[l_u].end(),
                                 make_pair(0, threshold), wcmp);
            pull_volume += distance(adj_short[l_u].begin(), r) * 2;
        }
    }

    ll l_diff = push_volume - pull_volume, g_diff;
    MPI_Allreduce(&l_diff, &g_diff, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return g_diff < 0; // push < pull
}
// }}}
void exchange_requests( // {{{
    const std::vector<std::vector<Request>> &send_requests,
    std::vector<std::vector<Request>> &recv_requests)
{
    std::vector<int> send_counts(world_size);
    for (int i = 0; i < world_size; i++) {
        send_counts[i] = send_requests[i].size();
    }

    std::vector<int> recv_counts(world_size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    std::vector<int> send_displs(world_size + 1, 0);
    std::vector<int> recv_displs(world_size + 1, 0);
    for (int i = 0; i < world_size; i++) {
        send_displs[i + 1] = send_displs[i] + send_counts[i];
        recv_displs[i + 1] = recv_displs[i] + recv_counts[i];
    }

    int total_send = send_displs[world_size];
    int total_recv = recv_displs[world_size];
    std::vector<Request> send_buffer(total_send);
    std::vector<Request> recv_buffer(total_recv);

    for (int i = 0; i < world_size; i++) {
        std::copy(send_requests[i].begin(), send_requests[i].end(),
                  send_buffer.begin() + send_displs[i]);
    }

    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(),
                  MPI_REQUEST_TYPE, recv_buffer.data(), recv_counts.data(),
                  recv_displs.data(), MPI_REQUEST_TYPE, MPI_COMM_WORLD);

    for (int i = 0; i < world_size; i++) {
        int start = recv_displs[i];
        int end = recv_displs[i + 1];
        recv_requests[i].assign(recv_buffer.begin() + start,
                                recv_buffer.begin() + end);
    }
}
// }}}
void pull_long_edges(vector<vector<pll>> &send_responses) // {{{
{
    vector<vector<Request>> send_requests(world_size);
    vector<vector<Request>> recv_requests(world_size);

    const auto &adj_long = adj_remote_long;
    for (auto it = B.upper_bound(k); it != B.end(); ++it) {
        for (ll u : it->second) {
            int l_u = u - v_first;
            ll threshold = d[l_u] - k * Delta;
            for (const auto &[v, w] : adj_long[l_u]) {
                if (w >= threshold)
                    break;
                send_requests[get_owner(v)].push_back({u, v, w});
            }

            if constexpr (!ENABLE_IOS)
                continue;

            const auto &adj_short = adj_remote_short;
            for (const auto &[v, w] : adj_short[l_u]) {
                if (w >= threshold)
                    break;
                send_requests[get_owner(v)].push_back({u, v, w});
            }
        }
    }

    exchange_requests(send_requests, recv_requests);

    for (int rank = 0; rank < world_size; rank++) {
        for (const auto &req : recv_requests[rank]) {
            ll t = req.target;
            ll s = req.source;
            ll w = req.weight;

            if (!is_local(s)) // skip if not local
                continue;

            ll bucket = d[s - v_first] / Delta;
            if (bucket == k)
                send_responses[rank].emplace_back(t, d[s - v_first] + w);
        }
    }
}
// }}}
void process_bucket(ll k) // {{{
{
    set<ll> A;
    vector<vector<pll>> send_updates(world_size);

    if (B.count(k))
        A = B[k]; // active vertices

    while (true) {
        relax<true, true>(A, send_updates);
        relax<true, false>(A, send_updates);
        communicate_updates(send_updates);

        A = std::move(updated); // active vertices
        updated.clear();

        int g_stop, l_stop = A.empty();
        MPI_Allreduce(&l_stop, &g_stop, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (g_stop)
            break;
    }

    relax<false, true>(B[k], send_updates);

    if constexpr (ENABLE_PUSH_PULL_PRUNING) {
        if (push_vs_pull())
            relax<false, false>(B[k], send_updates);
        else
            pull_long_edges(send_updates);
    } else {
        relax<false, false>(B[k], send_updates);
    }
    communicate_updates(send_updates);
}
// }}}
void delta_stepping() // {{{
{
    for (auto &e : adj_local_short)
        sort(e.begin(), e.end(), wcmp);
    for (auto &e : adj_local_long)
        sort(e.begin(), e.end(), wcmp);
    for (auto &e : adj_remote_short)
        sort(e.begin(), e.end(), wcmp);
    for (auto &e : adj_remote_long)
        sort(e.begin(), e.end(), wcmp);

    // d(rt) <- 0
    if (is_local(0))
        update_local(0, 0);
    for (int i = 0; i < v_cnt; i++)
        B[INF / Delta].insert(i + v_first);

    while (k != INF) {
        process_bucket(k);

        auto it = B.upper_bound(k);
        ll l = it == B.end() ? INF : it->first;
        if (l == INF / Delta)
            l = INF;
        MPI_Allreduce(&l, &k, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);

        if constexpr (ENABLE_HYBRIDIZATION) {
            ll l_processed = 0, g_processed;
            for (auto jt = B.begin(); jt != it; ++jt) {
                l_processed += jt->second.size();
                B.erase(jt);
            }
            MPI_Allreduce(&l_processed, &g_processed, 1, MPI_LONG_LONG, MPI_SUM,
                          MPI_COMM_WORLD);
            hybridization.processed += g_processed;

            if (hybridization.processed < hybridization.threshold)
                continue;

            set<ll> A;
            for (auto &[_, vertices] : B)
                A.insert(vertices.begin(), vertices.end());
            B.clear();
            B[0] = std::move(A);
            k = 0;
            Delta = INF; // switch to Bellman-Ford

            for (int u = 0; u < v_cnt; u++) {
                adj_local_short[u].insert(adj_local_short[u].end(),
                                          adj_local_long[u].begin(),
                                          adj_local_long[u].end());
                adj_local_long[u].clear();
                adj_remote_short[u].insert(adj_remote_short[u].end(),
                                           adj_remote_long[u].begin(),
                                           adj_remote_long[u].end());
                adj_remote_long[u].clear();
            }

            process_bucket(k);
            return;
        }
    }
}
// }}}
int main(int argc, char *argv[]) // {{{
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 3) {
        if (my_rank == 0)
            cerr << "Usage: " << argv[0] << " input_file output_file\n";
        MPI_Finalize();
        return 1;
    }

    string input_file = argv[1];
    string output_file = argv[2];

    load_graph(input_file);
    compute_boundaries();

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_LONG_LONG, MPI_LONG_LONG};
    MPI_Aint displacements[2];
    pll dummy;
    MPI_Get_address(&dummy.first, &displacements[0]);
    MPI_Get_address(&dummy.second, &displacements[1]);
    displacements[1] -= displacements[0];
    displacements[0] = 0;
    MPI_Type_create_struct(2, blocklengths, displacements, types,
                           &MPI_UPDATE_TYPE);
    MPI_Type_commit(&MPI_UPDATE_TYPE);

    int blocklengths_req[3] = {1, 1, 1};
    MPI_Datatype types_req[3] = {MPI_LONG_LONG, MPI_LONG_LONG, MPI_LONG_LONG};
    MPI_Aint displacements_req[3];
    Request dummy_req;
    MPI_Get_address(&dummy_req.target, &displacements_req[0]);
    MPI_Get_address(&dummy_req.source, &displacements_req[1]);
    MPI_Get_address(&dummy_req.weight, &displacements_req[2]);
    for (int i = 2; i >= 0; i--)
        displacements_req[i] -= displacements_req[0];
    MPI_Type_create_struct(3, blocklengths_req, displacements_req, types_req,
                           &MPI_REQUEST_TYPE);
    MPI_Type_commit(&MPI_REQUEST_TYPE);

    // TODO: load balancing
    delta_stepping();

    double end_time = MPI_Wtime();

    write_output(output_file);
    MPI_Type_free(&MPI_UPDATE_TYPE);
    MPI_Type_free(&MPI_REQUEST_TYPE);

    double l_time = end_time - start_time, g_time;
    MPI_Reduce(&l_time, &g_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
#ifdef DELTA
        Delta = DELTA;
#else
        Delta = 25;
#endif
        cerr << world_size << "," << n << "," << Delta << "," << ENABLE_IOS
             << "," << ENABLE_PUSH_PULL_PRUNING << "," << ENABLE_HYBRIDIZATION
             << "," << HYBRIDIZATION_TAU << "," << ENABLE_LOAD_BALANCING << ","
             << g_time << endl;
    }

    MPI_Finalize();
    return 0;
}
// }}}
