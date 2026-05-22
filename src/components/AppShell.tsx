import { type ReactNode } from 'react';

interface AppShellProps {
  toolbar: ReactNode;
  sidebar: ReactNode;
  detail: ReactNode;
}

export function AppShell({ toolbar, sidebar, detail }: AppShellProps) {
  return (
    <div className="h-full flex flex-col">
      <div className="h-12 bg-bg-surface border-b border-border-default flex items-center px-4 gap-3">
        {toolbar}
      </div>
      <div className="flex-1 flex overflow-hidden">
        <div className="w-[260px] bg-bg-surface border-r border-border-default flex flex-col overflow-hidden">
          {sidebar}
        </div>
        <div className="flex-1 bg-bg-primary overflow-auto">
          {detail}
        </div>
      </div>
    </div>
  );
}
