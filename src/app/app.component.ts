import { Component, OnInit } from '@angular/core';
import { DataService } from './data.service';
import { SummaryData } from './models/summary.model';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'SUM';
  data: SummaryData;
  loading = false;
  error: string | null = null;

  constructor(private dataService: DataService) { }

  ngOnInit() {
    this.loading = true;
    this.dataService.getData().subscribe({
      next: (data) => {
        this.data = data;
        this.loading = false;
      },
      error: (error) => {
        this.error = 'Failed to load data';
        this.loading = false;
      }
    });
  }
}
