//+------------------------------------------------------------------+
//| CalendarExporter.mq5                                              |
//| Exports the MT5 built-in economic calendar to a CSV that the      |
//| Python bot reads (news features + news-window rules).             |
//|                                                                   |
//| Attach to any chart. It writes once on start, then refreshes      |
//| every RefreshMinutes. Output goes to the COMMON files folder:     |
//|   %APPDATA%\MetaQuotes\Terminal\Common\Files\calendar.csv         |
//| Times are written in MT5 SERVER time, same as the price bars.     |
//+------------------------------------------------------------------+
#property copyright "PropBot"
#property version   "1.00"
#property strict

input int    DaysBack       = 3300;   // ~9 years of history for training
input int    DaysForward    = 14;     // upcoming events for live news windows
input int    RefreshMinutes = 10;
input string FileName       = "calendar.csv";

int OnInit()
  {
   ExportCalendar();
   EventSetTimer(RefreshMinutes * 60);
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason) { EventKillTimer(); }
void OnTimer()                  { ExportCalendar(); }
void OnTick()                   {}

string Fmt(datetime t)
  {
   MqlDateTime d; TimeToStruct(t, d);
   return StringFormat("%04d-%02d-%02d %02d:%02d:00", d.year, d.mon, d.day, d.hour, d.min);
  }

string Clean(string s)
  {
   StringReplace(s, ",", " ");
   StringReplace(s, "\"", "");
   return s;
  }

void ExportCalendar()
  {
   datetime from = TimeTradeServer() - (datetime)DaysBack * 86400;
   datetime to   = TimeTradeServer() + (datetime)DaysForward * 86400;

   MqlCalendarValue values[];
   int n = CalendarValueHistory(values, from, to);
   if(n <= 0)
     {
      PrintFormat("CalendarExporter: no values (%d), error %d", n, GetLastError());
      return;
     }

   string tmp = FileName + ".tmp";
   int h = FileOpen(tmp, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
     {
      PrintFormat("CalendarExporter: cannot open file, error %d", GetLastError());
      return;
     }
   FileWriteString(h, "time,currency,impact,event,actual,forecast,previous\r\n");

   int written = 0;
   for(int i = 0; i < n; i++)
     {
      MqlCalendarEvent ev;
      if(!CalendarEventById(values[i].event_id, ev)) continue;
      MqlCalendarCountry c;
      if(!CalendarCountryById(ev.country_id, c)) continue;

      string act = values[i].HasActualValue()   ? DoubleToString(values[i].GetActualValue(), 6)   : "";
      string fc  = values[i].HasForecastValue() ? DoubleToString(values[i].GetForecastValue(), 6) : "";
      string pv  = values[i].HasPreviousValue() ? DoubleToString(values[i].GetPreviousValue(), 6) : "";

      FileWriteString(h, StringFormat("%s,%s,%d,%s,%s,%s,%s\r\n",
                      Fmt(values[i].time), c.currency, (int)ev.importance,
                      Clean(ev.name), act, fc, pv));
      written++;
     }
   FileClose(h);

   // replace atomically so Python never reads a half-written file
   FileDelete(FileName, FILE_COMMON);
   if(!FileMove(tmp, FILE_COMMON, FileName, FILE_COMMON | FILE_REWRITE))
      PrintFormat("CalendarExporter: rename failed, error %d", GetLastError());
   else
      PrintFormat("CalendarExporter: wrote %d events", written);
  }
//+------------------------------------------------------------------+
