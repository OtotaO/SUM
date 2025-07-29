import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { SummaryData } from './models/summary.model';

@Injectable({
  providedIn: 'root'
})
export class DataService {
  constructor() { }

  getData(): Observable<SummaryData> {
    // Mock data for demonstration
    const data: SummaryData = {
      total_cases: 1000,
      recovered: 900,
      deaths: 100
    };
    return of(data);
  }
}
