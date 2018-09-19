from performance_comparison_template import draw

# got data
# label = ["DL2", "DRF", "Optimus"]
# poisson distribution
poisson_jct_times = [2.611,3.458, 2.700]  # value of bars, change here
poisson_makespan_times = [29.120,32.860, 31.240]

draw(poisson_jct_times, poisson_makespan_times, './performance_comparison_poisson_workload.pdf')



# google trace distribution
google_trace_jct_times = [2.726, 3.307, 2.847]  # value of bars, change here
google_trace_makespan_times = [27.060, 29.720, 28.240]

draw(google_trace_jct_times, google_trace_makespan_times, './performance_comparison_google_trace_workload.pdf')

