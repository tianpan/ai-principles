interface SummaryBoxProps {
  whatYouSee: string;
  whatItMeans: string;
  codeSnippet?: string;
}

export default function SummaryBox({ whatYouSee, whatItMeans, codeSnippet }: SummaryBoxProps) {
  return (
    <div className="highlight-card my-6">
      <div className="flex items-start gap-4">
        <div className="text-cyan-400 text-2xl">💡</div>
        <div className="flex-1">
          <div className="mb-4">
            <h4 className="text-cyan-400 font-bold mb-2">我看到了什么</h4>
            <p className="text-slate-300">{whatYouSee}</p>
          </div>
          <div>
            <h4 className="text-green-400 font-bold mb-2">这意味着什么</h4>
            <p className="text-slate-300">{whatItMeans}</p>
          </div>
          {codeSnippet && (
            <div className="mt-4">
              <pre className="text-xs bg-slate-900 p-3 rounded-lg overflow-x-auto">
                <code>{codeSnippet}</code>
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
