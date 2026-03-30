import { useState } from 'react';
import type { ChatFilters } from '../services/chatService';

interface FilterBarProps {
  onFiltersChange: (filters: ChatFilters) => void;
}

const PUBLICATION_TYPES = [
  'Meta-Analysis',
  'Systematic Review',
  'Randomized Controlled Trial',
  'Clinical Trial',
  'Review',
  'Case Reports',
];

export default function FilterBar({ onFiltersChange }: FilterBarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [yearMin, setYearMin] = useState('');
  const [yearMax, setYearMax] = useState('');
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);
  const [humanOnly, setHumanOnly] = useState(false);

  const applyFilters = () => {
    const filters: ChatFilters = {};
    if (yearMin) filters.year_min = parseInt(yearMin);
    if (yearMax) filters.year_max = parseInt(yearMax);
    if (selectedTypes.length > 0) filters.publication_types = selectedTypes;
    if (humanOnly) filters.human_only = true;
    onFiltersChange(filters);
  };

  const clearFilters = () => {
    setYearMin('');
    setYearMax('');
    setSelectedTypes([]);
    setHumanOnly(false);
    onFiltersChange({});
  };

  const toggleType = (type: string) => {
    setSelectedTypes(prev =>
      prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type]
    );
  };

  const hasActiveFilters = yearMin || yearMax || selectedTypes.length > 0 || humanOnly;

  return (
    <div className="filter-bar">
      <button className="filter-bar-toggle" onClick={() => setIsOpen(!isOpen)}>
        <span className="filter-icon">⚙</span>
        <span>Filters</span>
        {hasActiveFilters && <span className="filter-active-dot" />}
      </button>
      {isOpen && (
        <div className="filter-bar-panel">
          <div className="filter-group">
            <label className="filter-label">Year Range</label>
            <div className="filter-year-inputs">
              <input
                type="number"
                placeholder="From"
                value={yearMin}
                onChange={e => setYearMin(e.target.value)}
                className="filter-input"
                min="2000"
                max="2030"
              />
              <span className="filter-dash">–</span>
              <input
                type="number"
                placeholder="To"
                value={yearMax}
                onChange={e => setYearMax(e.target.value)}
                className="filter-input"
                min="2000"
                max="2030"
              />
            </div>
          </div>

          <div className="filter-group">
            <label className="filter-label">Study Type</label>
            <div className="filter-chips">
              {PUBLICATION_TYPES.map(type => (
                <button
                  key={type}
                  className={`filter-chip ${selectedTypes.includes(type) ? 'active' : ''}`}
                  onClick={() => toggleType(type)}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          <div className="filter-group">
            <label className="filter-checkbox-label">
              <input
                type="checkbox"
                checked={humanOnly}
                onChange={e => setHumanOnly(e.target.checked)}
              />
              Human studies only
            </label>
          </div>

          <div className="filter-actions">
            <button className="filter-apply-btn" onClick={applyFilters}>Apply</button>
            {hasActiveFilters && (
              <button className="filter-clear-btn" onClick={clearFilters}>Clear</button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
