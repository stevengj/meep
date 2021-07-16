/* Copyright (C) 2005-2021 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software Foundation,
%  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

#include "meep.hpp"

#include <algorithm>
#include <iterator>
#include <map>

using namespace std;

namespace meep {

namespace {

constexpr size_t kMeepTimingStackSize = 10;

const std::map<time_sink, const char *> kDescriptionByTimeSink{
    {Stepping, "time stepping"},
    {Connecting, "connecting chunks"},
    {Boundaries, "copying boundaries"},
    {MpiAllTime, "all-all communication"},
    {MpiOneTime, "1-1 communication"},
    {FieldOutput, "outputting fields"},
    {FourierTransforming, "Fourier transforming"},
    {MPBTime, "MPB mode solver"},
    {GetFarfieldsTime, "far-field transform"},
    {FieldUpdateB, "updating B field"},
    {FieldUpdateH, "updating H field"},
    {FieldUpdateD, "updating D field"},
    {FieldUpdateE, "updating E field"},
    {BoundarySteppingB, "boundary stepping B"},
    {BoundarySteppingWH, "boundary stepping WH"},
    {BoundarySteppingPH, "boundary stepping PH"},
    {BoundarySteppingH, "boundary stepping H"},
    {BoundarySteppingD, "boundary stepping D"},
    {BoundarySteppingWE, "boundary stepping WE"},
    {BoundarySteppingPE, "boundary stepping PE"},
    {BoundarySteppingE, "boundary stepping E"},
    {Other, "everything else"},
};

std::vector<double> timing_data_vector(const std::unordered_map<time_sink, double> &timers) {
  std::vector<double> ret;
  for (const auto &desc_ts : kDescriptionByTimeSink) {
    auto it = timers.find(desc_ts.first);
    ret.push_back((it != timers.end()) ? it->second : 0.);
  }
  return ret;
}

std::vector<double> timing_data_vector_from_all(const std::unordered_map<time_sink, double> &timers) {
  std::vector<double> time_spent_vector = timing_data_vector(timers);
  const int n = count_processors();
  std::vector<double> alltimes_tmp(n * time_spent_vector.size());
  for (size_t i = 0; i < time_spent_vector.size(); ++i) {
    alltimes_tmp[i * n + my_rank()] = time_spent_vector[i];
  }

  std::vector<double> alltimes(alltimes_tmp.size());
  sum_to_all(alltimes_tmp.data(), alltimes.data(), alltimes_tmp.size());
  return alltimes;
}

void pt(double mean, double stddev, const char *label) {
  if (mean != 0) {
    if (stddev != 0)
      master_printf("    %21s: %g s +/- %g s\n", label, mean, stddev);
    else
      master_printf("    %21s: %g s\n", label, mean);
  }
}

} // namespace

timing_scope::timing_scope(std::unordered_map<time_sink, double> *timers_, time_sink sink_)
    : timers(timers_), sink(sink_), active(true), t_start(wall_time()) {}

timing_scope::~timing_scope() { exit(); }

void timing_scope::exit() {
  if (!active) return;
  (*timers)[sink] += (wall_time() - t_start);
  active = false;
}

timing_scope fields::with_timing_scope(time_sink sink) { return timing_scope(&times_spent, sink); }

void fields::finished_working() {
  if (!was_working_on.empty()) { was_working_on.pop_back(); }
  working_on =
      with_timing_scope(!was_working_on.empty() ? was_working_on.back() : Other);
}

void fields::am_now_working_on(time_sink sink) {
  working_on = with_timing_scope(sink);
  was_working_on.push_back(sink);
  if (was_working_on.size() > kMeepTimingStackSize) {
    was_working_on.erase(was_working_on.begin());
  }
}

void fields::reset_timers() {
  was_working_on.clear();
  am_now_working_on(Other);
  times_spent.clear();
}

double fields::get_time_spent_on(time_sink sink) const {
  const auto it = times_spent.find(sink);
  return (it != times_spent.end()) ? it->second : 0.;
}

std::vector<double> fields::time_spent_on(time_sink sink) {
  int n = count_processors();
  std::vector<double> time_spent_per_process(n), temp(n);
  temp[my_rank()] = get_time_spent_on(sink);
  sum_to_all(&temp[0], &time_spent_per_process[0], n);
  return time_spent_per_process;
}

double fields::mean_time_spent_on(time_sink s) {
  int n = count_processors();
  double total_time_spent = sum_to_all(get_time_spent_on(s));
  return total_time_spent / n;
}

void fields::print_times() {
  std::vector<double> time_spent_vector = timing_data_vector(times_spent);
  std::vector<double> square_times;
  std::transform(time_spent_vector.begin(), time_spent_vector.end(),
                 std::back_inserter(square_times), [](double t) -> double { return t * t; });

  std::vector<double> mean(time_spent_vector.size());
  std::vector<double> stddev(time_spent_vector.size());
  sum_to_master(time_spent_vector.data(), mean.data(), time_spent_vector.size());
  sum_to_master(square_times.data(), stddev.data(), time_spent_vector.size());

  const int n = count_processors();
  for (size_t i = 0; i < time_spent_vector.size(); ++i) {
    mean[i] /= n;
    stddev[i] -= n * mean[i] * mean[i];
    stddev[i] = (n == 1 || stddev[i] <= 0) ? 0.0 : sqrt(stddev[i] / (n - 1));
  }

  master_printf("\nField time usage:\n");
  ptrdiff_t i = 0;
  for (const auto &desc_ts : kDescriptionByTimeSink) {
    pt(mean[i], stddev[i], desc_ts.second);
    ++i;
  }
  master_printf("\n");

  if (verbosity > 1) {
    master_printf("\nField time usage for all processes:\n");
    std::vector<double> alltimes = timing_data_vector_from_all(times_spent);

    int i = 0;
    for (const auto &desc_ts : kDescriptionByTimeSink) {
      master_printf("    %21s: %g", desc_ts.second, alltimes[i * n]);
      for (int j = 1; j < n; ++j)
        master_printf(", %g", alltimes[i * n + j]);
      master_printf("\n");
      ++i;
    }
    master_printf("\n");
  }
}

void fields::output_times(const char *fname) {
  if (verbosity > 0) master_printf("outputting timing statistics to file \"%s\"...\n", fname);
  FILE *tf = master_fopen(fname, "w");
  if (!tf) meep::abort("Unable to create file %s!\n", fname);

  std::vector<double> alltimes = timing_data_vector_from_all(times_spent);

  const char *sep = "";
  for (const auto &desc_ts : kDescriptionByTimeSink) {
    master_fprintf(tf, "%s%s", sep, desc_ts.second);
    sep = ", ";
  }
  master_fprintf(tf, "\n");

  const int n = count_processors();
  for (int j = 0; j < n; ++j) {
    const char *sep = "";
    for (size_t i = 0; i < kDescriptionByTimeSink.size(); ++i) {
      master_fprintf(tf, "%s%g", sep, alltimes[i * n + j]);
      sep = ", ";
    }
    master_fprintf(tf, "\n");
  }
  master_fclose(tf);
}

std::unordered_map<time_sink, std::vector<double> > fields::get_timing_data() const {
  std::vector<double> all_times = timing_data_vector_from_all(times_spent);
  const int n_procs = count_processors();

  std::unordered_map<time_sink, std::vector<double> > times_by_sink;
  auto it = all_times.begin();
  for (const auto &desc_ts : kDescriptionByTimeSink) {
    times_by_sink.emplace(std::make_pair(desc_ts.first, std::vector<double>(it, it + n_procs)));
    it += n_procs;
  }
  return times_by_sink;
}

} // namespace meep
