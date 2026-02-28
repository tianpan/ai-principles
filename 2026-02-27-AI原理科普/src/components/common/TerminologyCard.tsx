import { useState } from 'react';

interface TerminologyCardProps {
  term: string;
  definition: string;
  example?: string;
}

export default function TerminologyCard({ term, definition, example }: TerminologyCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="term-card">
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className="font-bold text-amber-400">{term}</span>
        <span className="text-slate-400">{isExpanded ? '−' : '+'}</span>
      </div>
      {isExpanded && (
        <div className="mt-2">
          <p className="text-slate-300 text-sm">{definition}</p>
          {example && (
            <p className="mt-2 text-slate-400 text-xs italic">示例：{example}</p>
          )}
        </div>
      )}
    </div>
  );
}
