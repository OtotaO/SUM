import { Component, OnInit, Input } from '@angular/core';
import { SummaryData } from '../models/summary.model';

@Component({
  selector: 'app-summary',
  templateUrl: './summary.component.html',
  styleUrls: ['./summary.component.css']
})
export class SummaryComponent implements OnInit {
  @Input() data: SummaryData;
  @Input() loading = false;
  @Input() error: string | null = null;

  constructor() { }

  ngOnInit(): void {
  }

}
