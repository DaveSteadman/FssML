using System.Diagnostics;

public class RunTimer
{
    private Stopwatch _stopwatch;

    // Constructor starts the timer immediately
    public RunTimer()
    {
        _stopwatch = Stopwatch.StartNew();
    }

    // Property to retrieve elapsed time in milliseconds as a float
    public float ElapsedMilliseconds => _stopwatch.ElapsedMilliseconds;

    // Property to retrieve elapsed time in seconds.
    public float ElapsedSeconds => (float)_stopwatch.Elapsed.TotalSeconds;
}