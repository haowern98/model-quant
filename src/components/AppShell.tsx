import { type ReactNode } from 'react';

interface AppShellProps {
  toolbar: ReactNode;
  sidebar: ReactNode;
  detail: ReactNode;
}

export function AppShell({ toolbar, sidebar, detail }: AppShellProps) {
  return (
    <div className="h-full min-h-0 flex flex-col">
      <div className="h-12 shrink-0 bg-bg-surface border-b border-border-default flex items-center px-4 gap-3">
        {toolbar}
      </div>
      <div className="flex-1 min-h-0 flex overflow-hidden">
        <div className="w-[260px] bg-bg-surface border-r border-border-default flex flex-col overflow-hidden">
          {sidebar}
        </div>
        <div className="flex-1 min-w-0 min-h-0 bg-bg-primary overflow-hidden">
          {detail}
        </div>
      </div>
    </div>
  );
}
