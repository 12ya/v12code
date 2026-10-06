import {
  type EnvironmentId,
  type MessageId,
  type ScopedThreadRef,
  type ServerProviderSkill,
  type TurnId,
} from "@v12code/contracts";
import { parseScopedThreadKey } from "@v12code/client-runtime/environment";
import { resolveChatListAnchoredEndSpace } from "@v12code/shared/chatList";
import {
  createContext,
  Fragment,
  memo,
  use,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
  type RefObject,
} from "react";
import { flushSync } from "react-dom";
import { LegendList, type LegendListRef } from "@legendapp/list/react";
import { FileDiff } from "@pierre/diffs/react";
import {
  deriveTimelineEntries,
  formatDuration,
  workEntryIndicatesToolFailure,
  workEntryIndicatesToolNeutralStatus,
  workEntryIndicatesToolSuccess,
  workLogEntryIsToolLike,
} from "../../session-logic";
import { type TurnDiffSummary } from "../../types";
import { summarizeTurnDiffStats } from "../../lib/turnDiffTree";
import {
  getRenderablePatch,
  resolveDiffThemeName,
  resolveFileDiffPath,
} from "../../lib/diffRendering";
import ChatMarkdown from "../ChatMarkdown";
import {
  BotIcon,
  CheckIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  CircleAlertIcon,
  EyeIcon,
  GitForkIcon,
  GlobeIcon,
  HammerIcon,
  MessageCircleIcon,
  MessageSquareIcon,
  MousePointerClickIcon,
  PaintbrushIcon,
  MinusIcon,
  SquarePenIcon,
  TerminalIcon,
  Trash2Icon,
  WrenchIcon,
  XIcon,
  ZapIcon,
} from "lucide-react";
import { Button } from "../ui/button";
import { buildExpandedImagePreview, ExpandedImagePreview } from "./ExpandedImagePreview";
import { ProposedPlanCard } from "./ProposedPlanCard";
import { ChangedFilesTree } from "./ChangedFilesTree";
import { DiffStatLabel, hasNonZeroStat } from "./DiffStatLabel";
import { MessageCopyButton } from "./MessageCopyButton";
import {
  computeStableMessagesTimelineRows,
  deriveMessagesTimelineRows,
  normalizeCompactToolLabel,
  resolveAssistantMessageCopyState,
  resolveTimelineIsAtEnd,
  resolveTimelineScrollThumb,
  type StableMessagesTimelineRowsState,
  type MessagesTimelineRow,
  type TimelineLatestTurn,
} from "./MessagesTimeline.logic";
import { TerminalContextInlineChip } from "./TerminalContextInlineChip";
import { Tooltip, TooltipPopup, TooltipTrigger } from "../ui/tooltip";
import { Popover, PopoverPopup } from "../ui/popover";
import { Collapsible, CollapsiblePanel } from "../ui/collapsible";
import {
  deriveDisplayedUserMessageState,
  type ParsedTerminalContextEntry,
} from "~/lib/terminalContext";
import {
  extractTrailingElementContexts,
  type ParsedElementContextEntry,
} from "~/lib/elementContext";
import {
  extractTrailingPreviewAnnotation,
  type ParsedPreviewAnnotation,
} from "~/lib/previewAnnotation";
import { cn } from "~/lib/utils";
import { useUiStateStore } from "~/uiStateStore";
import { useMeasuredScrollHeight } from "~/hooks/useMeasuredScrollHeight";
import { type TimestampFormat } from "@v12code/contracts/settings";
import { formatChatTimestampTooltip, formatShortTimestamp } from "../../timestampFormat";
import {
  extractTrailingTaskAnnotations,
  filterPendingContextTasks,
  type ContextualTask,
  useTaskHudStore,
} from "../../taskHudState";

import {
  buildInlineTerminalContextText,
  formatInlineTerminalContextLabel,
  textContainsInlineTerminalContextLabels,
} from "./userMessageTerminalContexts";
import { SkillInlineText } from "./SkillInlineText";
import { formatWorkspaceRelativePath } from "../../filePathDisplay";
import {
  buildReviewCommentRenderablePatch,
  formatReviewCommentFence,
  parseReviewCommentMessageSegments,
  type ReviewCommentContext,
} from "../../reviewCommentContext";

// ---------------------------------------------------------------------------
// Context — shared state consumed by every row component via Context.
// Propagates through LegendList's memo boundaries for shared callbacks and
// non-row-scoped state. `nowIso` is intentionally excluded — self-ticking
// components (WorkingTimer, LiveElapsed) handle it.
// ---------------------------------------------------------------------------

interface TimelineRowSharedState {
  timestampFormat: TimestampFormat;
  routeThreadKey: string;
  threadRef: ScopedThreadRef | null;
  markdownCwd: string | undefined;
  resolvedTheme: "light" | "dark";
  workspaceRoot: string | undefined;
  skills: ReadonlyArray<Pick<ServerProviderSkill, "name" | "displayName">>;
  activeThreadEnvironmentId: EnvironmentId;
  anchorMessageId: MessageId | null;
  sourceHighlightMessageId: MessageId | null;
  contextTasksBySourceMessageId: ReadonlyMap<MessageId, readonly ContextualTask[]>;
  contextTaskOrdinalById: ReadonlyMap<string, number>;
  onRevertUserMessage: (messageId: MessageId) => void;
  onForkMessage: (messageId: MessageId, initialPrompt?: string) => void;
  onAddSelectionTask: (input: {
    messageId: MessageId;
    author: "user" | "assistant";
    createdAt: string;
    quote: string;
    instruction: string;
  }) => void;
  onImageExpand: (preview: ExpandedImagePreview) => void;
  onOpenTurnDiff: (turnId: TurnId, filePath?: string) => void;
  onToggleTurnFold: (turnId: TurnId) => void;
  onToggleWorkGroup: (groupId: string, anchorElement?: HTMLElement) => void;
  onFoldAnimationStart: () => void;
}

interface TimelineRowActivityState {
  isWorking: boolean;
  isRevertingCheckpoint: boolean;
  activeTurnInProgress: boolean;
}

const TimelineRowCtx = createContext<TimelineRowSharedState>(null!);
const TimelineRowActivityCtx = createContext<TimelineRowActivityState>(null!);
const TIMELINE_LIST_HEADER = <div className="h-3 sm:h-4" />;
const TIMELINE_LIST_FOOTER = <div className="h-3 sm:h-4" />;
const EMPTY_TIMELINE_SKILLS: ReadonlyArray<Pick<ServerProviderSkill, "name" | "displayName">> = [];
const EMPTY_CONTEXT_TASKS: readonly ContextualTask[] = Object.freeze([]);
const NOOP_ADD_SELECTION_TASK: TimelineRowSharedState["onAddSelectionTask"] = () => {};
const TURN_FOLD_ANIMATION_MS = 200;

// ---------------------------------------------------------------------------
// Props (public API)
// ---------------------------------------------------------------------------

interface MessagesTimelineProps {
  isWorking: boolean;
  activeTurnInProgress: boolean;
  activeTurnStartedAt: string | null;
  listRef: React.RefObject<LegendListRef | null>;
  timelineEntries: ReturnType<typeof deriveTimelineEntries>;
  latestTurn: TimelineLatestTurn | null;
  runningTurnId: TurnId | null;
  turnDiffSummaryByAssistantMessageId: Map<MessageId, TurnDiffSummary>;
  routeThreadKey: string;
  onOpenTurnDiff: (turnId: TurnId, filePath?: string) => void;
  revertTurnCountByUserMessageId: Map<MessageId, number>;
  onRevertUserMessage: (messageId: MessageId) => void;
  onForkMessage: (messageId: MessageId, initialPrompt?: string) => void;
  onAddSelectionTask?: TimelineRowSharedState["onAddSelectionTask"];
  isRevertingCheckpoint: boolean;
  onImageExpand: (preview: ExpandedImagePreview) => void;
  activeThreadEnvironmentId: EnvironmentId;
  markdownCwd: string | undefined;
  resolvedTheme: "light" | "dark";
  timestampFormat: TimestampFormat;
  workspaceRoot: string | undefined;
  skills?: ReadonlyArray<Pick<ServerProviderSkill, "name" | "displayName">>;
  anchorMessageId: MessageId | null;
  sourceHighlightMessageId?: MessageId | null;
  onAnchorReady: (messageId: MessageId, anchorIndex: number) => void;
  onAnchorSizeChanged: (messageId: MessageId, size: number) => void;
  contentInsetEndAdjustment: number;
  onIsAtEndChange: (isAtEnd: boolean) => void;
}

// ---------------------------------------------------------------------------
// MessagesTimeline — list owner
// ---------------------------------------------------------------------------

export const MessagesTimeline = memo(function MessagesTimeline({
  isWorking,
  activeTurnInProgress,
  activeTurnStartedAt,
  listRef,
  timelineEntries,
  latestTurn,
  runningTurnId,
  turnDiffSummaryByAssistantMessageId,
  routeThreadKey,
  onOpenTurnDiff,
  revertTurnCountByUserMessageId,
  onRevertUserMessage,
  onForkMessage,
  onAddSelectionTask = NOOP_ADD_SELECTION_TASK,
  isRevertingCheckpoint,
  onImageExpand,
  activeThreadEnvironmentId,
  markdownCwd,
  resolvedTheme,
  timestampFormat,
  workspaceRoot,
  skills = EMPTY_TIMELINE_SKILLS,
  anchorMessageId,
  sourceHighlightMessageId = null,
  onAnchorReady,
  onAnchorSizeChanged,
  contentInsetEndAdjustment,
  onIsAtEndChange,
}: MessagesTimelineProps) {
  const storedContextTasks = useTaskHudStore(
    (state) => state.contextTasksByThreadKey[routeThreadKey] ?? EMPTY_CONTEXT_TASKS,
  );
  const contextTasks = useMemo(
    () => filterPendingContextTasks(storedContextTasks),
    [storedContextTasks],
  );
  const contextTasksBySourceMessageId = useMemo(
    () => groupContextTasksBySourceMessageId(contextTasks),
    [contextTasks],
  );
  const contextTaskOrdinalById = useMemo(
    () => new Map(contextTasks.map((task, index) => [task.id, index + 1] as const)),
    [contextTasks],
  );
  const [expandedTurnIds, setExpandedTurnIds] = useState<ReadonlySet<TurnId>>(new Set());
  const [collapsingTurnIds, setCollapsingTurnIds] = useState<ReadonlySet<TurnId>>(new Set());
  const expandedTurnIdsRef = useRef(expandedTurnIds);
  const collapsingTurnIdsRef = useRef(collapsingTurnIds);
  const turnFoldTimersRef = useRef(new Map<TurnId, ReturnType<typeof setTimeout>>());
  const foldAnimationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [foldAnimationActive, setFoldAnimationActive] = useState(false);
  const [expandedWorkGroupIds, setExpandedWorkGroupIds] = useState<ReadonlySet<string>>(new Set());

  const updateExpandedTurnIds = useCallback(
    (update: (existing: ReadonlySet<TurnId>) => ReadonlySet<TurnId>) => {
      const next = update(expandedTurnIdsRef.current);
      expandedTurnIdsRef.current = next;
      setExpandedTurnIds(next);
    },
    [],
  );
  const updateCollapsingTurnIds = useCallback(
    (update: (existing: ReadonlySet<TurnId>) => ReadonlySet<TurnId>) => {
      const next = update(collapsingTurnIdsRef.current);
      collapsingTurnIdsRef.current = next;
      setCollapsingTurnIds(next);
    },
    [],
  );
  const onFoldAnimationStart = useCallback(() => {
    if (foldAnimationTimerRef.current) clearTimeout(foldAnimationTimerRef.current);
    setFoldAnimationActive(true);
    foldAnimationTimerRef.current = setTimeout(() => {
      foldAnimationTimerRef.current = null;
      setFoldAnimationActive(false);
    }, TURN_FOLD_ANIMATION_MS + 50);
  }, []);
  const onToggleTurnFold = useCallback(
    (turnId: TurnId) => {
      const pendingTimer = turnFoldTimersRef.current.get(turnId);
      if (pendingTimer) {
        clearTimeout(pendingTimer);
        turnFoldTimersRef.current.delete(turnId);
      }

      if (collapsingTurnIdsRef.current.has(turnId)) {
        updateCollapsingTurnIds((existing) => {
          const next = new Set(existing);
          next.delete(turnId);
          return next;
        });
        return;
      }

      if (!expandedTurnIdsRef.current.has(turnId)) {
        updateExpandedTurnIds((existing) => {
          const next = new Set(existing);
          next.add(turnId);
          return next;
        });
        return;
      }

      updateCollapsingTurnIds((existing) => {
        const next = new Set(existing);
        next.add(turnId);
        return next;
      });
      const timer = setTimeout(() => {
        turnFoldTimersRef.current.delete(turnId);
        updateExpandedTurnIds((existing) => {
          const next = new Set(existing);
          next.delete(turnId);
          return next;
        });
        updateCollapsingTurnIds((existing) => {
          const next = new Set(existing);
          next.delete(turnId);
          return next;
        });
      }, TURN_FOLD_ANIMATION_MS);
      turnFoldTimersRef.current.set(turnId, timer);
    },
    [updateCollapsingTurnIds, updateExpandedTurnIds],
  );
  useEffect(
    () => () => {
      for (const timer of turnFoldTimersRef.current.values()) clearTimeout(timer);
      if (foldAnimationTimerRef.current) clearTimeout(foldAnimationTimerRef.current);
    },
    [],
  );
  const onToggleWorkGroup = useCallback(
    (groupId: string, anchorElement?: HTMLElement) => {
      const anchorBottomBeforeToggle = anchorElement?.getBoundingClientRect().bottom ?? null;

      flushSync(() => {
        setExpandedWorkGroupIds((existing) => {
          const next = new Set(existing);
          if (next.has(groupId)) {
            next.delete(groupId);
          } else {
            next.add(groupId);
          }
          return next;
        });
      });

      if (anchorBottomBeforeToggle === null || !anchorElement) {
        return;
      }

      const delta = anchorElement.getBoundingClientRect().bottom - anchorBottomBeforeToggle;
      if (Math.abs(delta) < 0.5) {
        return;
      }

      const list = listRef.current;
      const currentScroll = list?.getState?.().scroll;
      if (list && typeof currentScroll === "number") {
        list.scrollToOffset({ offset: currentScroll + delta, animated: false });
      }
    },
    [listRef],
  );

  // An in-session interrupt leaves its turn expanded so the user keeps their
  // place; the next turn (or a reload, since this is local state) folds it.
  const previousLatestTurnRef = useRef(latestTurn);
  useEffect(() => {
    const previous = previousLatestTurnRef.current;
    previousLatestTurnRef.current = latestTurn;
    if (!latestTurn || previous?.turnId === undefined) {
      return;
    }
    if (latestTurn.turnId === previous.turnId) {
      if (previous.state === "running" && latestTurn.state === "interrupted") {
        updateExpandedTurnIds((existing) => {
          const next = new Set(existing);
          next.add(latestTurn.turnId);
          return next;
        });
      }
      return;
    }
    updateExpandedTurnIds((existing) => {
      if (!existing.has(previous.turnId)) {
        return existing;
      }
      const next = new Set(existing);
      next.delete(previous.turnId);
      return next;
    });
  }, [latestTurn, updateExpandedTurnIds]);

  const rawRows = useMemo(
    () =>
      deriveMessagesTimelineRows({
        timelineEntries,
        latestTurn,
        runningTurnId,
        expandedTurnIds,
        collapsingTurnIds,
        expandedWorkGroupIds,
        isWorking,
        activeTurnStartedAt,
        turnDiffSummaryByAssistantMessageId,
        revertTurnCountByUserMessageId,
      }),
    [
      timelineEntries,
      latestTurn,
      runningTurnId,
      expandedTurnIds,
      collapsingTurnIds,
      expandedWorkGroupIds,
      isWorking,
      activeTurnStartedAt,
      turnDiffSummaryByAssistantMessageId,
      revertTurnCountByUserMessageId,
    ],
  );
  const rows = useStableRows(rawRows);
  const scrollIndicatorTrackRef = useRef<HTMLDivElement>(null);
  const scrollIndicatorThumbRef = useRef<HTMLDivElement>(null);
  const handleAnchorReady = useCallback(
    (info: { anchorIndex: number | undefined }) => {
      if (anchorMessageId !== null && info.anchorIndex !== undefined) {
        onAnchorReady(anchorMessageId, info.anchorIndex);
      }
    },
    [anchorMessageId, onAnchorReady],
  );
  const handleAnchorSizeChanged = useCallback(
    (size: number) => {
      if (anchorMessageId !== null) {
        onAnchorSizeChanged(anchorMessageId, size);
      }
    },
    [anchorMessageId, onAnchorSizeChanged],
  );
  const anchoredEndSpace = useMemo(() => {
    const config = resolveChatListAnchoredEndSpace(rows, anchorMessageId, (row) =>
      row.kind === "message" ? row.message.id : null,
    );
    return config
      ? { ...config, onReady: handleAnchorReady, onSizeChanged: handleAnchorSizeChanged }
      : undefined;
  }, [anchorMessageId, handleAnchorReady, handleAnchorSizeChanged, rows]);

  const updateScrollIndicator = useCallback(() => {
    const state = listRef.current?.getState?.();
    const track = scrollIndicatorTrackRef.current;
    const thumb = scrollIndicatorThumbRef.current;
    if (!state || !track || !thumb || state.data.length === 0) {
      if (thumb) thumb.style.opacity = "0";
      return;
    }

    const lastIndex = state.data.length - 1;
    const lastTop = state.positionAtIndex(lastIndex);
    const lastHeight = state.sizeAtIndex(lastIndex);
    if (
      typeof lastTop !== "number" ||
      typeof lastHeight !== "number" ||
      !Number.isFinite(lastTop) ||
      !Number.isFinite(lastHeight)
    ) {
      thumb.style.opacity = "0";
      return;
    }

    const geometry = resolveTimelineScrollThumb({
      contentLength: lastTop + Math.max(1, lastHeight),
      scroll: state.scroll ?? 0,
      trackLength: track.getBoundingClientRect().height,
      viewportLength: state.scrollLength ?? 0,
    });
    if (!geometry) {
      thumb.style.opacity = "0";
      return;
    }

    thumb.style.height = `${geometry.length}px`;
    thumb.style.transform = `translateY(${geometry.offset}px)`;
    thumb.style.opacity = "1";
  }, [listRef]);

  const handleScroll = useCallback(() => {
    const state = listRef.current?.getState?.();
    const isAtEnd = resolveTimelineIsAtEnd(state);
    if (isAtEnd !== undefined) {
      onIsAtEndChange(isAtEnd);
    }
    updateScrollIndicator();
  }, [listRef, onIsAtEndChange, updateScrollIndicator]);

  useEffect(() => {
    const frame = requestAnimationFrame(handleScroll);
    return () => cancelAnimationFrame(frame);
  }, [handleScroll, rows.length]);

  useEffect(() => {
    const track = scrollIndicatorTrackRef.current;
    if (!track) return;
    const observer = new ResizeObserver(updateScrollIndicator);
    observer.observe(track);
    return () => observer.disconnect();
  }, [updateScrollIndicator]);

  const sharedState = useMemo<TimelineRowSharedState>(
    () => ({
      timestampFormat,
      routeThreadKey,
      threadRef: parseScopedThreadKey(routeThreadKey),
      markdownCwd,
      resolvedTheme,
      workspaceRoot,
      skills,
      activeThreadEnvironmentId,
      anchorMessageId,
      sourceHighlightMessageId,
      contextTasksBySourceMessageId,
      contextTaskOrdinalById,
      onRevertUserMessage,
      onForkMessage,
      onAddSelectionTask,
      onImageExpand,
      onOpenTurnDiff,
      onToggleTurnFold,
      onToggleWorkGroup,
      onFoldAnimationStart,
    }),
    [
      timestampFormat,
      routeThreadKey,
      markdownCwd,
      resolvedTheme,
      workspaceRoot,
      skills,
      activeThreadEnvironmentId,
      anchorMessageId,
      sourceHighlightMessageId,
      contextTasksBySourceMessageId,
      contextTaskOrdinalById,
      onRevertUserMessage,
      onForkMessage,
      onAddSelectionTask,
      onImageExpand,
      onOpenTurnDiff,
      onToggleTurnFold,
      onToggleWorkGroup,
      onFoldAnimationStart,
    ],
  );
  const activityState = useMemo<TimelineRowActivityState>(
    () => ({
      isWorking,
      isRevertingCheckpoint,
      activeTurnInProgress,
    }),
    [activeTurnInProgress, isRevertingCheckpoint, isWorking],
  );

  // Stable renderItem — no closure deps. Row components read shared state
  // from TimelineRowCtx, which propagates through LegendList's memo.
  const renderItem = useCallback(
    ({ item }: { item: MessagesTimelineRow }) => (
      <div className="mx-auto w-full min-w-0 max-w-3xl overflow-x-clip" data-timeline-root="true">
        <TimelineRowContent row={item} />
      </div>
    ),
    [],
  );

  if (rows.length === 0 && !isWorking) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-muted-foreground/30">
          Send a message to start the conversation.
        </p>
      </div>
    );
  }

  return (
    <TimelineRowCtx value={sharedState}>
      <TimelineRowActivityCtx value={activityState}>
        <div className="relative h-full min-h-0">
          <LegendList<MessagesTimelineRow>
            ref={listRef}
            data={rows}
            keyExtractor={keyExtractor}
            getItemType={getItemType}
            renderItem={renderItem}
            estimatedItemSize={90}
            initialScrollAtEnd
            {...(anchoredEndSpace ? { anchoredEndSpace } : {})}
            contentInsetEndAdjustment={contentInsetEndAdjustment}
            maintainScrollAtEnd={
              anchoredEndSpace || foldAnimationActive
                ? false
                : {
                    animated: false,
                    on: {
                      dataChange: true,
                      itemLayout: true,
                      layout: true,
                    },
                  }
            }
            maintainVisibleContentPosition={{
              data: true,
              size: !anchoredEndSpace && !foldAnimationActive,
            }}
            onScroll={handleScroll}
            className="timeline-scroll-viewport h-full min-h-0 overflow-x-hidden overscroll-y-contain px-3 [overflow-anchor:none] sm:px-5"
            ListHeaderComponent={TIMELINE_LIST_HEADER}
            ListFooterComponent={TIMELINE_LIST_FOOTER}
          />
          <div
            aria-hidden="true"
            className="pointer-events-none absolute top-1 z-50 w-[var(--app-scrollbar-width)]"
            data-testid="timeline-scroll-indicator"
            ref={scrollIndicatorTrackRef}
            style={{
              bottom: Math.max(4, Math.ceil(contentInsetEndAdjustment) + 4),
              right: 0,
            }}
          >
            <div
              className="absolute right-0 top-0 w-full rounded-full bg-muted-foreground/35 opacity-0 transition-opacity duration-150"
              ref={scrollIndicatorThumbRef}
            />
          </div>
        </div>
      </TimelineRowActivityCtx>
    </TimelineRowCtx>
  );
});

function keyExtractor(item: MessagesTimelineRow) {
  return item.id;
}

function getItemType(item: MessagesTimelineRow) {
  return item.kind === "message" ? `message:${item.message.role}` : item.kind;
}

// ---------------------------------------------------------------------------
// TimelineRowContent — the actual row component
// ---------------------------------------------------------------------------

type TimelineEntry = ReturnType<typeof deriveTimelineEntries>[number];
type TimelineMessage = Extract<TimelineEntry, { kind: "message" }>["message"];
type TimelineWorkEntry = Extract<MessagesTimelineRow, { kind: "work" }>["groupedEntries"][number];
type TimelineRow = MessagesTimelineRow;

const TimelineRowContent = memo(function TimelineRowContent({ row }: { row: TimelineRow }) {
  const content = (
    <div
      className={cn(
        // Commentary (non-terminal assistant) rows carry no metadata row, so
        // they sit closer to the work that follows them.
        (row.kind === "message" && row.message.role === "assistant" && !row.showAssistantMeta) ||
          row.kind === "work" ||
          row.kind === "work-toggle"
          ? "pb-2"
          : "pb-4",
        row.kind === "message" && row.message.role === "assistant" ? "group/assistant" : null,
      )}
    >
      {row.kind === "work" ? <WorkGroupSection groupedEntries={row.groupedEntries} /> : null}
      {row.kind === "work-toggle" ? <WorkGroupToggleTimelineRow row={row} /> : null}
      {row.kind === "turn-fold" ? <TurnFoldTimelineRow row={row} /> : null}
      {row.kind === "message" && row.message.role === "user" ? <UserTimelineRow row={row} /> : null}
      {row.kind === "message" && row.message.role === "assistant" ? (
        <AssistantTimelineRow row={row} />
      ) : null}
      {row.kind === "proposed-plan" ? <ProposedPlanTimelineRow row={row} /> : null}
      {row.kind === "working" ? <WorkingTimelineRow row={row} /> : null}
    </div>
  );

  return (
    <div
      data-timeline-row-id={row.id}
      data-timeline-row-kind={row.kind}
      data-message-id={row.kind === "message" ? row.message.id : undefined}
      data-message-role={row.kind === "message" ? row.message.role : undefined}
    >
      {row.foldState ? (
        <FoldedTimelineRowTransition closing={row.foldState === "closing"}>
          {content}
        </FoldedTimelineRowTransition>
      ) : (
        content
      )}
    </div>
  );
});

function FoldedTimelineRowTransition({
  children,
  closing,
}: {
  children: ReactNode;
  closing: boolean;
}) {
  const [entered, setEntered] = useState(false);

  useEffect(() => {
    const frame = requestAnimationFrame(() => setEntered(true));
    return () => cancelAnimationFrame(frame);
  }, []);

  const visible = entered && !closing;
  return (
    <div
      aria-hidden={!visible}
      className={cn(
        "grid transition-[grid-template-rows,opacity] duration-200 ease-out motion-reduce:transition-none",
        visible ? "grid-rows-[1fr] opacity-100" : "pointer-events-none grid-rows-[0fr] opacity-0",
      )}
    >
      <div className="min-h-0 overflow-hidden">{children}</div>
    </div>
  );
}

interface TaskSourceAnchor {
  readonly key: string;
  readonly ordinals: readonly number[];
  readonly left: number;
  readonly top: number;
}

export function groupContextTaskOrdinalsByQuote(
  tasks: readonly ContextualTask[],
  ordinalByTaskId: ReadonlyMap<string, number>,
): ReadonlyMap<string, readonly number[]> {
  const groups = new Map<string, number[]>();
  for (const task of tasks) {
    const ordinal = ordinalByTaskId.get(task.id);
    if (ordinal === undefined) continue;
    const quoteOrdinals = groups.get(task.quote);
    if (quoteOrdinals) {
      quoteOrdinals.push(ordinal);
    } else {
      groups.set(task.quote, [ordinal]);
    }
  }
  return groups;
}

function collectTaskSourceTextNodes(root: Node): Text[] {
  const nodes: Text[] = [];
  for (const child of root.childNodes) {
    if (child.nodeType === 3) {
      const textNode = child as Text;
      if (!textNode.parentElement?.closest("[data-task-source-marker]")) {
        nodes.push(textNode);
      }
      continue;
    }
    if (child instanceof HTMLElement && child.dataset.taskSourceMarker !== undefined) {
      continue;
    }
    nodes.push(...collectTaskSourceTextNodes(child));
  }
  return nodes;
}

function measureTaskSourceAnchor(
  container: HTMLElement,
  quote: string,
): Pick<TaskSourceAnchor, "left" | "top"> | null {
  const textNodes = collectTaskSourceTextNodes(container);
  const fullText = textNodes.map((node) => node.data).join("");
  const quoteStart = fullText.indexOf(quote);
  if (quoteStart < 0) return null;

  const quoteEnd = quoteStart + quote.length;
  let traversed = 0;
  let startNode: Text | null = null;
  let startOffset = 0;
  let endNode: Text | null = null;
  let endOffset = 0;

  for (const node of textNodes) {
    const nodeEnd = traversed + node.data.length;
    if (!startNode && quoteStart >= traversed && quoteStart <= nodeEnd) {
      startNode = node;
      startOffset = quoteStart - traversed;
    }
    if (quoteEnd >= traversed && quoteEnd <= nodeEnd) {
      endNode = node;
      endOffset = quoteEnd - traversed;
      break;
    }
    traversed = nodeEnd;
  }

  if (!startNode || !endNode) return null;
  const range = document.createRange();
  range.setStart(startNode, startOffset);
  range.setEnd(endNode, endOffset);
  const rects = Array.from(range.getClientRects()).filter(
    (rect) => rect.width > 0 || rect.height > 0,
  );
  const quoteRect = rects.at(-1) ?? range.getBoundingClientRect();
  if (quoteRect.width === 0 && quoteRect.height === 0) return null;

  const containerRect = container.getBoundingClientRect();
  return {
    left: Math.min(Math.max(quoteRect.right - containerRect.left, 10), containerRect.width - 10),
    top: quoteRect.bottom - containerRect.top,
  };
}

function TaskSourceMarkersOverlay({
  messageId,
  containerRef,
  contentKey,
}: {
  readonly messageId: MessageId;
  readonly containerRef: RefObject<HTMLDivElement | null>;
  readonly contentKey: string;
}) {
  const ctx = use(TimelineRowCtx);
  const tasks = ctx.contextTasksBySourceMessageId.get(messageId) ?? EMPTY_CONTEXT_TASKS;
  const taskGroups = useMemo(
    () => groupContextTaskOrdinalsByQuote(tasks, ctx.contextTaskOrdinalById),
    [ctx.contextTaskOrdinalById, tasks],
  );
  const [anchors, setAnchors] = useState<readonly TaskSourceAnchor[]>([]);
  const [editingTask, setEditingTask] = useState<{
    readonly task: ContextualTask;
    readonly anchorRect: SelectionAnchorRect;
    readonly markerLabel: string;
  } | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || taskGroups.size === 0) {
      setAnchors([]);
      return;
    }

    const measure = () => {
      const next: TaskSourceAnchor[] = [];
      for (const [quote, ordinals] of taskGroups) {
        const position = measureTaskSourceAnchor(container, quote);
        if (position) next.push({ key: quote, ordinals, ...position });
      }
      setAnchors((current) => {
        if (
          current.length === next.length &&
          current.every(
            (anchor, index) =>
              anchor.key === next[index]?.key &&
              anchor.ordinals.join(",") === next[index]?.ordinals.join(",") &&
              Math.abs(anchor.left - (next[index]?.left ?? 0)) < 0.5 &&
              Math.abs(anchor.top - (next[index]?.top ?? 0)) < 0.5,
          )
        ) {
          return current;
        }
        return next;
      });
    };

    const frame = requestAnimationFrame(measure);
    const observer = new ResizeObserver(measure);
    observer.observe(container);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [containerRef, contentKey, taskGroups]);

  if (anchors.length === 0) return null;

  const highlighted = ctx.sourceHighlightMessageId === messageId;
  return (
    <>
      {anchors.map((anchor) => (
        <button
          key={anchor.key}
          type="button"
          aria-label={`Task ${anchor.ordinals.join(", ")} from this selection`}
          data-source-highlight={highlighted ? "true" : undefined}
          data-task-source-marker="true"
          className={cn(
            "absolute z-10 flex h-5 min-w-5 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 border-background bg-blue-500 px-1 font-semibold text-[10px] text-white shadow-sm transition-[scale,box-shadow] hover:scale-110",
            highlighted ? "scale-110 shadow-[0_0_0_4px_rgb(59_130_246/0.2)]" : null,
          )}
          style={{ left: anchor.left, top: anchor.top }}
          onPointerUp={(event) => event.stopPropagation()}
          onClick={(event) => {
            event.stopPropagation();
            const task = tasks.find((candidate) => candidate.quote === anchor.key);
            if (!task) return;
            setEditingTask({
              task,
              anchorRect: event.currentTarget.getBoundingClientRect(),
              markerLabel: String(ctx.contextTaskOrdinalById.get(task.id) ?? 1),
            });
          }}
        >
          {anchor.ordinals.join(",")}
        </button>
      ))}
      {editingTask ? (
        <SelectionActionBar
          quote={editingTask.task.quote}
          anchorRect={editingTask.anchorRect}
          initialInstruction={
            editingTask.task.instruction === editingTask.task.quote
              ? ""
              : editingTask.task.instruction
          }
          initiallyEditing
          markerLabel={editingTask.markerLabel}
          onCancel={() => setEditingTask(null)}
          onSave={(instruction) => {
            useTaskHudStore
              .getState()
              .setContextTaskInstruction(ctx.routeThreadKey, editingTask.task.id, instruction);
            setEditingTask(null);
          }}
        />
      ) : null}
    </>
  );
}

function UserTimelineRow({ row }: { row: Extract<TimelineRow, { kind: "message" }> }) {
  const ctx = use(TimelineRowCtx);
  const {
    selectionContainerRef,
    selectedQuote,
    selectionAnchorRect,
    clearSelection,
    captureSelection,
  } = useMessageTaskSelection();
  const userImages = row.message.attachments ?? [];
  const taskAnnotations = extractTrailingTaskAnnotations(row.message.text);
  const displayedUserMessage = deriveDisplayedUserMessageState(taskAnnotations.promptText);
  const terminalContexts = displayedUserMessage.contexts;
  const previewAnnotations: ParsedPreviewAnnotation[] = [];
  let visibleText = displayedUserMessage.visibleText;
  while (true) {
    const extracted = extractTrailingPreviewAnnotation(visibleText);
    if (!extracted.annotation) break;
    previewAnnotations.unshift(extracted.annotation);
    visibleText = extracted.promptText;
  }
  const elementContextState = extractTrailingElementContexts(visibleText);
  const elementContexts = [
    ...displayedUserMessage.elementContexts,
    ...elementContextState.contexts,
  ];
  const previewImages = userImages.filter((image) => image.name.startsWith("preview-annotation-"));
  const regularImages = userImages.filter((image) => !image.name.startsWith("preview-annotation-"));
  const canRevertAgentWork = typeof row.revertTurnCount === "number";
  const isAnnotationOnlyMessage =
    taskAnnotations.annotationCount > 0 &&
    regularImages.length === 0 &&
    previewAnnotations.length === 0 &&
    elementContexts.length === 0 &&
    terminalContexts.length === 0 &&
    elementContextState.promptText.trim().length === 0;

  return (
    <div className="group flex flex-col items-end gap-1">
      <div className="flex max-w-[80%] flex-col items-end gap-1.5">
        {taskAnnotations.annotationCount > 0 ? (
          <div className="inline-flex h-8 items-center gap-1.5 rounded-full border border-border bg-secondary px-2.5 text-xs font-medium text-foreground">
            <MessageSquareIcon className="size-3 text-muted-foreground" />
            <span>
              {taskAnnotations.annotationCount}{" "}
              {taskAnnotations.annotationCount === 1 ? "annotation" : "annotations"}
            </span>
          </div>
        ) : null}
        {!isAnnotationOnlyMessage ? (
          <div
            ref={selectionContainerRef}
            className="relative min-w-0 rounded-2xl border border-border bg-secondary p-3"
            onPointerUp={captureSelection}
            onKeyUp={captureSelection}
          >
            {regularImages.length > 0 && (
              <div className="mb-2 grid max-w-[420px] grid-cols-2 gap-2">
                {regularImages.map((image: NonNullable<TimelineMessage["attachments"]>[number]) => (
                  <div
                    key={image.id}
                    className="overflow-hidden rounded-lg border border-border/80 bg-background/70"
                  >
                    {image.previewUrl ? (
                      <button
                        type="button"
                        className="h-full w-full cursor-zoom-in"
                        aria-label={`Preview ${image.name}`}
                        onClick={() => {
                          const preview = buildExpandedImagePreview(regularImages, image.id);
                          if (!preview) return;
                          ctx.onImageExpand(preview);
                        }}
                      >
                        <img
                          src={image.previewUrl}
                          alt={image.name}
                          className="block h-auto max-h-[220px] w-full object-cover"
                        />
                      </button>
                    ) : (
                      <div className="flex min-h-[72px] items-center justify-center px-2 py-3 text-center text-[11px] text-muted-foreground/70">
                        {image.name}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
            {previewAnnotations.map((annotation, index) => (
              <UserMessagePreviewAnnotationCard
                key={annotation.id}
                annotation={annotation}
                image={previewImages[index] ?? null}
              />
            ))}
            {elementContexts.length > 0 ? (
              <div className="mb-2 flex flex-wrap gap-1.5">
                {elementContexts.map((context) => (
                  <UserMessageElementContextChip
                    key={`${context.header}:${context.body}`}
                    context={context}
                  />
                ))}
              </div>
            ) : null}
            <CollapsibleUserMessageBody
              text={elementContextState.promptText}
              terminalContexts={terminalContexts}
              skills={ctx.skills}
              markdownCwd={ctx.markdownCwd}
            />
            <TaskSourceMarkersOverlay
              messageId={row.message.id}
              containerRef={selectionContainerRef}
              contentKey={elementContextState.promptText}
            />
          </div>
        ) : null}
      </div>
      <div className="flex w-full max-w-[80%] items-center justify-end pe-1 text-xs tabular-nums opacity-0 transition-opacity duration-200 focus-within:opacity-100 group-hover:opacity-100 max-sm:opacity-100">
        <div className="flex shrink-0 items-center gap-2">
          <Tooltip>
            <TooltipTrigger render={<p className="text-muted-foreground text-xs tabular-nums" />}>
              {formatShortTimestamp(row.message.createdAt, ctx.timestampFormat)}
            </TooltipTrigger>
            <TooltipPopup>
              {formatChatTimestampTooltip(row.message.createdAt, ctx.timestampFormat)}
            </TooltipPopup>
          </Tooltip>
          <div className="flex items-center gap-0.5">
            <ForkMessageButton messageId={row.message.id} />
            {canRevertAgentWork && <RevertUserMessageButton messageId={row.message.id} />}
            {displayedUserMessage.copyText && (
              <MessageCopyButton text={displayedUserMessage.copyText} variant="ghost" />
            )}
          </div>
        </div>
      </div>
      {selectedQuote && selectionAnchorRect ? (
        <SelectionActionBar
          quote={selectedQuote}
          anchorRect={selectionAnchorRect}
          onCancel={clearSelection}
          onSave={(instruction) => {
            ctx.onAddSelectionTask({
              messageId: row.message.id,
              author: "user",
              createdAt: row.message.createdAt,
              quote: selectedQuote,
              instruction,
            });
            clearSelection();
          }}
        />
      ) : null}
    </div>
  );
}

function RevertUserMessageButton({ messageId }: { messageId: MessageId }) {
  const ctx = use(TimelineRowCtx);
  const activity = use(TimelineRowActivityCtx);

  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            type="button"
            size="xs"
            variant="ghost"
            className="gap-1 px-1.5"
            disabled={activity.isRevertingCheckpoint}
            onClick={() => ctx.onRevertUserMessage(messageId)}
            aria-label="Edit and rerun"
          />
        }
      >
        <SquarePenIcon className="size-3" />
        <span className="hidden sm:inline">Edit &amp; rerun</span>
      </TooltipTrigger>
      <TooltipPopup side="top">Edit and rerun</TooltipPopup>
    </Tooltip>
  );
}

function ForkMessageButton({ messageId }: { messageId: MessageId }) {
  const ctx = use(TimelineRowCtx);

  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            type="button"
            size="xs"
            variant="ghost"
            onClick={() => ctx.onForkMessage(messageId)}
            aria-label="Fork chat from here"
          />
        }
      >
        <GitForkIcon className="size-3" />
      </TooltipTrigger>
      <TooltipPopup side="top">Fork chat from here</TooltipPopup>
    </Tooltip>
  );
}

function TurnFoldTimelineRow({ row }: { row: Extract<TimelineRow, { kind: "turn-fold" }> }) {
  const ctx = use(TimelineRowCtx);

  return (
    <div className="border-b border-border/60 pb-2 pt-1">
      <button
        type="button"
        aria-expanded={row.expanded}
        data-scroll-anchor-ignore
        onClick={() => ctx.onToggleTurnFold(row.turnId)}
        className="flex cursor-pointer select-none items-center gap-1 rounded-md px-1 text-xs text-muted-foreground tabular-nums transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/70"
      >
        <span>{row.label}</span>
        <ChevronRightIcon
          className={cn(
            "size-3.5 transition-transform duration-200 ease-out motion-reduce:transition-none",
            row.expanded && "rotate-90",
          )}
        />
      </button>
    </div>
  );
}

function AssistantTimelineRow({ row }: { row: Extract<TimelineRow, { kind: "message" }> }) {
  const ctx = use(TimelineRowCtx);
  const {
    selectionContainerRef,
    selectedQuote,
    selectionAnchorRect,
    clearSelection,
    captureSelection,
  } = useMessageTaskSelection();
  const messageText = row.message.text || (row.message.streaming ? "" : "(empty response)");

  return (
    <>
      <div className="min-w-0">
        <div
          ref={selectionContainerRef}
          className="relative min-w-0 rounded-lg px-1 py-0.5"
          onPointerUp={captureSelection}
          onKeyUp={captureSelection}
        >
          <ChatMarkdown
            text={messageText}
            cwd={ctx.markdownCwd}
            threadRef={ctx.threadRef ?? undefined}
            isStreaming={Boolean(row.message.streaming)}
            skills={ctx.skills}
          />
          <AssistantChangedFilesSection
            turnSummary={row.assistantTurnDiffSummary}
            routeThreadKey={ctx.routeThreadKey}
            resolvedTheme={ctx.resolvedTheme}
            onOpenTurnDiff={ctx.onOpenTurnDiff}
          />
          {row.showAssistantMeta ? (
            <div className="mt-1.5 flex items-center gap-2 text-xs tabular-nums opacity-0 transition-opacity duration-200 focus-within:opacity-100 group-hover/assistant:opacity-100">
              <AssistantCopyButton row={row} />
              {!row.message.streaming ? <ForkMessageButton messageId={row.message.id} /> : null}
              {!row.message.streaming && (
                <Tooltip>
                  <TooltipTrigger
                    render={<p className="text-muted-foreground text-xs tabular-nums" />}
                  >
                    {formatShortTimestamp(row.message.updatedAt, ctx.timestampFormat)}
                  </TooltipTrigger>
                  <TooltipPopup>
                    {formatChatTimestampTooltip(row.message.updatedAt, ctx.timestampFormat)}
                  </TooltipPopup>
                </Tooltip>
              )}
            </div>
          ) : null}
          <TaskSourceMarkersOverlay
            messageId={row.message.id}
            containerRef={selectionContainerRef}
            contentKey={messageText}
          />
        </div>
      </div>
      {selectedQuote && selectionAnchorRect ? (
        <SelectionActionBar
          quote={selectedQuote}
          anchorRect={selectionAnchorRect}
          onCancel={clearSelection}
          onSave={(instruction) => {
            ctx.onAddSelectionTask({
              messageId: row.message.id,
              author: "assistant",
              createdAt: row.message.createdAt,
              quote: selectedQuote,
              instruction,
            });
            clearSelection();
          }}
        />
      ) : null}
    </>
  );
}

function useMessageTaskSelection() {
  const selectionContainerRef = useRef<HTMLDivElement>(null);
  const [selectedQuote, setSelectedQuote] = useState<string | null>(null);
  const [selectionAnchorRect, setSelectionAnchorRect] = useState<SelectionAnchorRect | null>(null);
  const selectionTimerRef = useRef<number | null>(null);
  const captureSelection = useCallback(() => {
    if (selectionTimerRef.current !== null) window.clearTimeout(selectionTimerRef.current);
    selectionTimerRef.current = window.setTimeout(() => {
      selectionTimerRef.current = null;
      const container = selectionContainerRef.current;
      if (!container) return;
      const selection = readSelectionWithin(container);
      if (selection) {
        setSelectedQuote(selection.quote);
        setSelectionAnchorRect(selection.anchorRect);
      }
    }, 0);
  }, []);

  useEffect(() => {
    return () => {
      if (selectionTimerRef.current !== null) window.clearTimeout(selectionTimerRef.current);
    };
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedQuote(null);
    setSelectionAnchorRect(null);
  }, []);

  return {
    selectionContainerRef,
    selectedQuote,
    selectionAnchorRect,
    clearSelection,
    captureSelection,
  };
}

interface SelectionAnchorRect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  readonly top: number;
  readonly right: number;
  readonly bottom: number;
  readonly left: number;
}

function readSelectionWithin(
  container: HTMLElement,
): { readonly quote: string; readonly anchorRect: SelectionAnchorRect } | null {
  const selection = window.getSelection();
  const quote = selection?.toString().trim() ?? "";
  if (!selection || selection.isCollapsed || quote.length === 0) return null;
  const anchor = selection.anchorNode;
  const focus = selection.focusNode;
  if (!anchor || !focus || !container.contains(anchor) || !container.contains(focus)) return null;
  const rangeRect = selection.getRangeAt(0).getBoundingClientRect();
  return {
    quote,
    anchorRect: {
      x: rangeRect.x,
      y: rangeRect.y,
      width: rangeRect.width,
      height: rangeRect.height,
      top: rangeRect.top,
      right: rangeRect.right,
      bottom: rangeRect.bottom,
      left: rangeRect.left,
    },
  };
}

function SelectionActionBar(props: {
  quote: string;
  anchorRect: SelectionAnchorRect;
  initialInstruction?: string;
  initiallyEditing?: boolean;
  markerLabel?: string;
  onSave: (instruction: string) => void;
  onCancel: () => void;
}) {
  const [instruction, setInstruction] = useState(props.initialInstruction ?? "");
  const [editingContext, setEditingContext] = useState(props.initiallyEditing ?? false);
  const instructionRef = useRef<HTMLTextAreaElement>(null);
  const virtualAnchor = useMemo(
    () => ({ getBoundingClientRect: () => props.anchorRect }),
    [props.anchorRect],
  );
  useEffect(() => {
    if (!editingContext) return;
    const frame = requestAnimationFrame(() => instructionRef.current?.focus());
    return () => cancelAnimationFrame(frame);
  }, [editingContext]);
  return (
    <Popover open onOpenChange={(open) => !open && props.onCancel()}>
      <PopoverPopup
        anchor={virtualAnchor}
        side={editingContext ? "right" : "top"}
        align="center"
        sideOffset={editingContext ? 14 : 6}
        positionerClassName="transition-none"
        className={cn(
          "max-w-[calc(100vw-1rem)] transition-none data-ending-style:scale-100 data-starting-style:scale-100 data-starting-style:opacity-100",
          editingContext ? "w-[min(19rem,calc(100vw-2rem))] rounded-2xl" : "w-auto rounded-md",
        )}
        viewportClassName="overflow-visible p-0 [--viewport-inline-padding:0px]"
      >
        <div onPointerUp={(event) => event.stopPropagation()}>
          {editingContext ? (
            <div className="relative flex min-h-32 flex-col p-3">
              <span className="absolute top-1/2 -left-8 flex size-6 -translate-y-1/2 items-center justify-center rounded-full border-2 border-white bg-blue-500 font-semibold text-[11px] text-white shadow-sm">
                {props.markerLabel ?? "1"}
              </span>
              <textarea
                autoFocus
                ref={instructionRef}
                value={instruction}
                onChange={(event) => setInstruction(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Escape") props.onCancel();
                  if (
                    shouldSaveSelectionTaskContextOnKeyDown({
                      key: event.key,
                      shiftKey: event.shiftKey,
                      isComposing: event.nativeEvent.isComposing,
                    })
                  ) {
                    event.preventDefault();
                    props.onSave(resolveSelectionTaskInstruction(props.quote, instruction));
                  }
                }}
                placeholder="Add an optional comment…"
                aria-label="Optional task context"
                className="min-h-20 w-full resize-none bg-transparent text-sm outline-none placeholder:text-muted-foreground/55"
              />
              <div className="mt-2 flex items-center justify-between">
                <Button
                  type="button"
                  size="icon-xs"
                  variant="ghost"
                  aria-label="Discard task context"
                  onClick={props.onCancel}
                >
                  <Trash2Icon className="size-3.5" />
                </Button>
                <div className="flex items-center gap-1.5">
                  <Button
                    type="button"
                    size="xs"
                    variant="outline"
                    className="rounded-full"
                    onClick={() => setEditingContext(false)}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="button"
                    size="xs"
                    className="rounded-full"
                    onClick={() =>
                      props.onSave(resolveSelectionTaskInstruction(props.quote, instruction))
                    }
                  >
                    Save
                  </Button>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex h-8 items-stretch whitespace-nowrap">
              <button
                type="button"
                className="rounded-md px-2.5 text-xs transition-colors hover:bg-accent"
                onClick={() => setEditingContext(true)}
              >
                Add to task
              </button>
            </div>
          )}
        </div>
      </PopoverPopup>
    </Popover>
  );
}

export function resolveSelectionTaskInstruction(quote: string, optionalContext: string): string {
  return optionalContext.trim() || quote;
}

export function groupContextTasksBySourceMessageId(
  tasks: readonly ContextualTask[],
): ReadonlyMap<MessageId, readonly ContextualTask[]> {
  const groups = new Map<MessageId, ContextualTask[]>();
  for (const task of tasks) {
    const messageTasks = groups.get(task.sourceMessageId);
    if (messageTasks) {
      messageTasks.push(task);
    } else {
      groups.set(task.sourceMessageId, [task]);
    }
  }
  return groups;
}

export function shouldSaveSelectionTaskContextOnKeyDown(input: {
  readonly key: string;
  readonly shiftKey: boolean;
  readonly isComposing: boolean;
}): boolean {
  return input.key === "Enter" && !input.shiftKey && !input.isComposing;
}

function AssistantCopyButton({ row }: { row: Extract<TimelineRow, { kind: "message" }> }) {
  const assistantCopyState = resolveAssistantMessageCopyState({
    text: row.message.text ?? null,
    showCopyButton: row.showAssistantCopyButton,
    streaming: row.assistantCopyStreaming,
  });

  if (!assistantCopyState.visible) {
    return null;
  }

  return <MessageCopyButton text={assistantCopyState.text ?? ""} variant="ghost" />;
}

function ProposedPlanTimelineRow({
  row,
}: {
  row: Extract<TimelineRow, { kind: "proposed-plan" }>;
}) {
  const ctx = use(TimelineRowCtx);

  return (
    <div className="min-w-0 px-1 py-0.5">
      <ProposedPlanCard
        planMarkdown={row.proposedPlan.planMarkdown}
        environmentId={ctx.activeThreadEnvironmentId}
        threadRef={ctx.threadRef ?? undefined}
        cwd={ctx.markdownCwd}
        workspaceRoot={ctx.workspaceRoot}
      />
    </div>
  );
}

function WorkingTimelineRow({ row }: { row: Extract<TimelineRow, { kind: "working" }> }) {
  return (
    <div
      className="border-b border-border/60 px-1 pb-2.5 pt-1"
      data-testid="working-status-separator"
    >
      <div className="text-sm text-muted-foreground tabular-nums">
        {row.createdAt ? (
          <>
            Working for <WorkingTimer createdAt={row.createdAt} />
          </>
        ) : (
          "Working..."
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Self-ticking labels — update their own text nodes so elapsed-time display
// does not create a React commit every second while a response is streaming.
// ---------------------------------------------------------------------------

/** Live "Working for Xs" label. */
function WorkingTimer({ createdAt }: { createdAt: string }) {
  const textRef = useRef<HTMLSpanElement>(null);
  const initialText = formatWorkingTimerNow(createdAt);

  useEffect(() => {
    const updateText = () => {
      if (textRef.current) {
        textRef.current.textContent = formatWorkingTimerNow(createdAt);
      }
    };
    updateText();
    const id = setInterval(updateText, 1000);
    return () => clearInterval(id);
  }, [createdAt]);

  return (
    <span ref={textRef} className="tabular-nums">
      {initialText}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Extracted row sections — own their state / store subscriptions so changes
// re-render only the affected row, not the entire list.
// ---------------------------------------------------------------------------

/** Renders one or more already-derived work log rows. Overflow expansion is modeled as LegendList data. */
const WorkGroupSection = memo(function WorkGroupSection({
  groupedEntries,
}: {
  groupedEntries: Extract<MessagesTimelineRow, { kind: "work" }>["groupedEntries"];
}) {
  const { workspaceRoot } = use(TimelineRowCtx);
  const nonEmptyEntries = useMemo(
    () => groupedEntries.filter((entry) => !workEntryIndicatesToolNeutralStatus(entry)),
    [groupedEntries],
  );
  const onlyToolEntries = nonEmptyEntries.every((entry) => workLogEntryIsToolLike(entry));
  const groupLabel = onlyToolEntries
    ? nonEmptyEntries.length === 1
      ? "1 tool call"
      : `${nonEmptyEntries.length} tool calls`
    : "Work Log";

  if (nonEmptyEntries.length === 0) return null;

  return (
    <section className="-mx-1 space-y-0.5 px-1 py-0.5" aria-label={groupLabel}>
      {!onlyToolEntries && (
        <p className="px-0.5 pb-0.5 font-medium text-[11px] text-muted-foreground/65">
          {groupLabel}
        </p>
      )}
      <div className="space-y-px">
        {nonEmptyEntries.map((workEntry) => (
          <SimpleWorkEntryRow
            key={workEntry.id}
            workEntry={workEntry}
            workspaceRoot={workspaceRoot}
          />
        ))}
      </div>
    </section>
  );
});

function WorkGroupToggleTimelineRow({
  row,
}: {
  row: Extract<TimelineRow, { kind: "work-toggle" }>;
}) {
  const ctx = use(TimelineRowCtx);
  return (
    <Collapsible open={row.expanded}>
      <button
        type="button"
        className="group flex w-full cursor-pointer items-center gap-1.5 rounded-md px-0.5 py-0.5 text-left text-[12px] leading-5 text-muted-foreground transition-colors duration-150 hover:text-foreground/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/70"
        aria-expanded={row.expanded}
        onClick={(event) => {
          const anchorElement =
            event.currentTarget.closest<HTMLElement>("[data-timeline-row-id]") ??
            event.currentTarget;
          ctx.onToggleWorkGroup(row.groupId, anchorElement);
        }}
      >
        <span className="flex size-5 shrink-0 items-center justify-center text-muted-foreground/75">
          <WrenchIcon className="size-3.5 stroke-[1.8]" aria-hidden />
        </span>
        <span className="min-w-0 truncate">{row.summary}</span>
        <ChevronRightIcon
          className={cn(
            "size-3 shrink-0 opacity-60 transition-transform duration-200 ease-out motion-reduce:transition-none",
            row.expanded && "rotate-90",
          )}
          aria-hidden
        />
      </button>
      <CollapsiblePanel className="transition-[height,opacity] ease-out data-ending-style:opacity-0 data-starting-style:opacity-0 motion-reduce:transition-none">
        <div className="pt-0.5">
          <WorkGroupSection groupedEntries={row.groupedEntries} />
        </div>
      </CollapsiblePanel>
    </Collapsible>
  );
}

/** Subscribes directly to the UI state store for expand/collapse state,
 *  so toggling re-renders only this component — not the entire list. */
const AssistantChangedFilesSection = memo(function AssistantChangedFilesSection({
  turnSummary,
  routeThreadKey,
  resolvedTheme,
  onOpenTurnDiff,
}: {
  turnSummary: TurnDiffSummary | undefined;
  routeThreadKey: string;
  resolvedTheme: "light" | "dark";
  onOpenTurnDiff: (turnId: TurnId, filePath?: string) => void;
}) {
  if (!turnSummary) return null;
  const checkpointFiles = turnSummary.files;
  if (checkpointFiles.length === 0) return null;

  return (
    <AssistantChangedFilesSectionInner
      turnSummary={turnSummary}
      checkpointFiles={checkpointFiles}
      routeThreadKey={routeThreadKey}
      resolvedTheme={resolvedTheme}
      onOpenTurnDiff={onOpenTurnDiff}
    />
  );
});

/** Inner component that only mounts when there are actual changed files,
 *  so the store subscription is unconditional (no hooks after early return). */
function AssistantChangedFilesSectionInner({
  turnSummary,
  checkpointFiles,
  routeThreadKey,
  resolvedTheme,
  onOpenTurnDiff,
}: {
  turnSummary: TurnDiffSummary;
  checkpointFiles: TurnDiffSummary["files"];
  routeThreadKey: string;
  resolvedTheme: "light" | "dark";
  onOpenTurnDiff: (turnId: TurnId, filePath?: string) => void;
}) {
  const allDirectoriesExpanded = useUiStateStore(
    (store) => store.threadChangedFilesExpandedById[routeThreadKey]?.[turnSummary.turnId] ?? true,
  );
  const setExpanded = useUiStateStore((store) => store.setThreadChangedFilesExpanded);
  const summaryStat = summarizeTurnDiffStats(checkpointFiles);
  const changedFileCountLabel = String(checkpointFiles.length);

  return (
    <div className="mt-2 rounded-lg border border-border/80 bg-card/45 p-2.5">
      <div className="sticky top-2 z-10 mb-1.5 flex items-center justify-between gap-2 bg-[color-mix(in_srgb,var(--card)_45%,var(--background))] before:absolute before:inset-x-0 before:-top-2 before:h-2 before:bg-[color-mix(in_srgb,var(--card)_45%,var(--background))] before:content-['']">
        <p className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground/65">
          <span>Changed files ({changedFileCountLabel})</span>
          {hasNonZeroStat(summaryStat) && (
            <>
              <span className="mx-1">•</span>
              <DiffStatLabel additions={summaryStat.additions} deletions={summaryStat.deletions} />
            </>
          )}
        </p>
        <div className="flex items-center gap-1.5">
          <Button
            type="button"
            size="xs"
            variant="outline"
            data-scroll-anchor-ignore
            onClick={() => setExpanded(routeThreadKey, turnSummary.turnId, !allDirectoriesExpanded)}
          >
            {allDirectoriesExpanded ? "Collapse all" : "Expand all"}
          </Button>
          <Button
            type="button"
            size="xs"
            variant="outline"
            onClick={() => onOpenTurnDiff(turnSummary.turnId, checkpointFiles[0]?.path)}
          >
            View diff
          </Button>
        </div>
      </div>
      <ChangedFilesTree
        key={`changed-files-tree:${turnSummary.turnId}`}
        turnId={turnSummary.turnId}
        files={checkpointFiles}
        allDirectoriesExpanded={allDirectoriesExpanded}
        resolvedTheme={resolvedTheme}
        onOpenTurnDiff={onOpenTurnDiff}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Leaf components
// ---------------------------------------------------------------------------

const UserMessageTerminalContextInlineLabel = memo(
  function UserMessageTerminalContextInlineLabel(props: { context: ParsedTerminalContextEntry }) {
    const tooltipText =
      props.context.body.length > 0
        ? `${props.context.header}\n${props.context.body}`
        : props.context.header;

    return <TerminalContextInlineChip label={props.context.header} tooltipText={tooltipText} />;
  },
);

const UserMessageElementContextChip = memo(function UserMessageElementContextChip(props: {
  context: ParsedElementContextEntry;
}) {
  const tooltipText = props.context.body
    ? `${props.context.header}\n${props.context.body}`
    : props.context.header;
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <span className="inline-flex max-w-full items-center gap-1 rounded-md border border-border/70 bg-background/70 px-1.5 py-0.5 text-xs text-foreground/85">
            <MousePointerClickIcon className="size-3 shrink-0" />
            <span className="truncate">{props.context.header}</span>
          </span>
        }
      />
      <TooltipPopup side="top" className="max-w-96 whitespace-pre-wrap leading-tight">
        {tooltipText}
      </TooltipPopup>
    </Tooltip>
  );
});

function UserMessagePreviewAnnotationCard(props: {
  annotation: ParsedPreviewAnnotation;
  image: NonNullable<TimelineMessage["attachments"]>[number] | null;
}) {
  const ctx = use(TimelineRowCtx);
  return (
    <div className="mb-2 flex max-w-full items-center overflow-hidden rounded-lg border border-border/70 bg-background/70">
      {props.image?.previewUrl ? (
        <button
          type="button"
          className="size-14 shrink-0 cursor-zoom-in overflow-hidden border-r border-border/70 bg-muted"
          aria-label={`Preview ${props.image.name}`}
          onClick={() => {
            if (!props.image) return;
            const preview = buildExpandedImagePreview([props.image], props.image.id);
            if (preview) ctx.onImageExpand(preview);
          }}
        >
          <img
            src={props.image.previewUrl}
            alt="Annotated preview crop"
            className="size-full object-cover"
          />
        </button>
      ) : null}
      <div className="min-w-0 px-2.5 py-2">
        {props.annotation.comment ? (
          <div className="max-w-80 truncate text-xs font-medium text-foreground/90">
            {props.annotation.comment}
          </div>
        ) : null}
        <div
          className={cn(
            "flex items-center gap-2 text-[10px] text-muted-foreground",
            props.annotation.comment && "mt-1",
          )}
        >
          {props.annotation.targetSummary ? (
            <span className="truncate">{props.annotation.targetSummary}</span>
          ) : null}
          {props.annotation.styleChanges.length > 0 ? (
            <span className="inline-flex shrink-0 items-center gap-1">
              <PaintbrushIcon className="size-3" />
              {props.annotation.styleChanges.length}
            </span>
          ) : null}
        </div>
      </div>
    </div>
  );
}

const MAX_COLLAPSED_USER_MESSAGE_LINES = 8;
const MAX_COLLAPSED_USER_MESSAGE_LENGTH = 600;
const COLLAPSED_USER_MESSAGE_FADE_HEIGHT_REM = 1.75;
const COLLAPSED_USER_MESSAGE_FADE_MASK = `linear-gradient(to bottom, black calc(100% - ${COLLAPSED_USER_MESSAGE_FADE_HEIGHT_REM}rem), transparent)`;

function shouldCollapseUserMessage(text: string): boolean {
  if (text.trim().length === 0) {
    return false;
  }

  return (
    text.length > MAX_COLLAPSED_USER_MESSAGE_LENGTH ||
    text.split("\n").length > MAX_COLLAPSED_USER_MESSAGE_LINES
  );
}

const CollapsibleUserMessageBody = memo(function CollapsibleUserMessageBody(props: {
  text: string;
  terminalContexts: ParsedTerminalContextEntry[];
  skills: ReadonlyArray<Pick<ServerProviderSkill, "name" | "displayName">>;
  markdownCwd: string | undefined;
  footer?: ReactNode;
}) {
  const [expanded, setExpanded] = useState(false);
  const hasVisibleBody = props.text.trim().length > 0 || props.terminalContexts.length > 0;
  const canCollapse = hasVisibleBody && shouldCollapseUserMessage(props.text);
  const isCollapsed = canCollapse && !expanded;
  const { ref: bodyRef, height: bodyHeight } = useMeasuredScrollHeight<HTMLDivElement>();

  return (
    <div>
      {hasVisibleBody ? (
        <div
          ref={bodyRef}
          className={cn(
            "relative",
            canCollapse &&
              "overflow-hidden transition-[max-height] duration-200 ease-out motion-reduce:transition-none",
          )}
          data-user-message-body="true"
          data-user-message-collapsed={isCollapsed ? "true" : "false"}
          data-user-message-collapsible={canCollapse ? "true" : "false"}
          data-user-message-fade={isCollapsed ? "true" : "false"}
          style={{
            maxHeight: canCollapse
              ? isCollapsed
                ? "11rem"
                : bodyHeight !== null
                  ? `${bodyHeight}px`
                  : undefined
              : undefined,
            WebkitMaskImage: isCollapsed ? COLLAPSED_USER_MESSAGE_FADE_MASK : undefined,
            maskImage: isCollapsed ? COLLAPSED_USER_MESSAGE_FADE_MASK : undefined,
          }}
        >
          <UserMessageBody
            text={props.text}
            terminalContexts={props.terminalContexts}
            skills={props.skills}
            markdownCwd={props.markdownCwd}
          />
        </div>
      ) : null}
      {canCollapse || props.footer ? (
        <div
          className={cn(
            "mt-1.5 flex items-center gap-2",
            canCollapse && props.footer ? "justify-between" : "justify-end",
          )}
          data-user-message-footer="true"
        >
          {canCollapse ? (
            <Button
              type="button"
              size="xs"
              variant="ghost"
              aria-expanded={expanded}
              data-scroll-anchor-ignore
              onClick={() => setExpanded((value) => !value)}
              className="-ml-1 h-6 rounded-md px-1.5 text-xs text-muted-foreground/72 hover:bg-muted/55 hover:text-foreground/85"
            >
              {expanded ? "Show less" : "Show full message"}
            </Button>
          ) : null}
          {props.footer ? (
            <div className="ml-auto flex items-center gap-2">{props.footer}</div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
});

const UserMessageBody = memo(function UserMessageBody(props: {
  text: string;
  terminalContexts: ParsedTerminalContextEntry[];
  skills: ReadonlyArray<Pick<ServerProviderSkill, "name" | "displayName">>;
  markdownCwd: string | undefined;
}) {
  const ctx = use(TimelineRowCtx);
  const renderInlineMarkdownSegment = (text: string, key: string) => {
    const leadingWhitespace = /^\s+/.exec(text)?.[0] ?? "";
    const textWithoutLeadingWhitespace = text.slice(leadingWhitespace.length);
    const trailingWhitespace = /\s+$/.exec(textWithoutLeadingWhitespace)?.[0] ?? "";
    const content = textWithoutLeadingWhitespace.slice(
      0,
      textWithoutLeadingWhitespace.length - trailingWhitespace.length,
    );

    return (
      <Fragment key={key}>
        {leadingWhitespace ? <span aria-hidden="true">{leadingWhitespace}</span> : null}
        {content ? (
          <ChatMarkdown
            text={content}
            cwd={props.markdownCwd}
            threadRef={ctx.threadRef ?? undefined}
            skills={props.skills}
            className="text-foreground"
            lineBreaks
          />
        ) : null}
        {trailingWhitespace ? <span aria-hidden="true">{trailingWhitespace}</span> : null}
      </Fragment>
    );
  };

  const reviewCommentSegments = parseReviewCommentMessageSegments(props.text);
  if (reviewCommentSegments.some((segment) => segment.kind === "review-comment")) {
    return (
      <div className="space-y-3 text-sm leading-relaxed text-foreground">
        {reviewCommentSegments.map((segment) =>
          segment.kind === "text" ? (
            segment.text.trim().length > 0 ? (
              <div key={segment.id} className="wrap-break-word">
                <ChatMarkdown
                  text={segment.text.trim()}
                  cwd={props.markdownCwd}
                  threadRef={ctx.threadRef ?? undefined}
                  skills={props.skills}
                  className="text-foreground"
                  lineBreaks
                />
              </div>
            ) : null
          ) : (
            <UserMessageReviewCommentCard key={segment.comment.id} comment={segment.comment} />
          ),
        )}
      </div>
    );
  }

  if (props.terminalContexts.length > 0) {
    const hasEmbeddedInlineLabels = textContainsInlineTerminalContextLabels(
      props.text,
      props.terminalContexts,
    );
    const inlinePrefix = buildInlineTerminalContextText(props.terminalContexts);
    const inlineNodes: ReactNode[] = [];

    if (hasEmbeddedInlineLabels) {
      let cursor = 0;

      for (const context of props.terminalContexts) {
        const label = formatInlineTerminalContextLabel(context.header);
        const matchIndex = props.text.indexOf(label, cursor);
        if (matchIndex === -1) {
          inlineNodes.length = 0;
          break;
        }
        if (matchIndex > cursor) {
          inlineNodes.push(
            renderInlineMarkdownSegment(
              props.text.slice(cursor, matchIndex),
              `user-terminal-context-inline-before:${context.header}:${cursor}`,
            ),
          );
        }
        inlineNodes.push(
          <UserMessageTerminalContextInlineLabel
            key={`user-terminal-context-inline:${context.header}`}
            context={context}
          />,
        );
        cursor = matchIndex + label.length;
      }

      if (inlineNodes.length > 0) {
        if (cursor < props.text.length) {
          inlineNodes.push(
            renderInlineMarkdownSegment(
              props.text.slice(cursor),
              `user-message-terminal-context-inline-rest:${cursor}`,
            ),
          );
        }

        return (
          <div className="whitespace-pre-wrap wrap-break-word text-sm leading-relaxed text-foreground">
            {inlineNodes}
          </div>
        );
      }
    }

    for (const context of props.terminalContexts) {
      inlineNodes.push(
        <UserMessageTerminalContextInlineLabel
          key={`user-terminal-context-inline:${context.header}`}
          context={context}
        />,
      );
      inlineNodes.push(
        <span key={`user-terminal-context-inline-space:${context.header}`} aria-hidden="true">
          {" "}
        </span>,
      );
    }

    if (props.text.length > 0) {
      inlineNodes.push(
        <ChatMarkdown
          key="user-message-terminal-context-inline-text"
          text={props.text}
          cwd={props.markdownCwd}
          threadRef={ctx.threadRef ?? undefined}
          skills={props.skills}
          className="text-foreground"
          lineBreaks
        />,
      );
    } else if (inlinePrefix.length === 0) {
      return null;
    }

    return (
      <div className="whitespace-pre-wrap wrap-break-word text-sm leading-relaxed text-foreground">
        {inlineNodes}
      </div>
    );
  }

  if (props.text.length === 0) {
    return null;
  }

  return (
    <ChatMarkdown
      text={props.text}
      cwd={props.markdownCwd}
      threadRef={ctx.threadRef ?? undefined}
      skills={props.skills}
      className="text-foreground"
      lineBreaks
    />
  );
});

function UserMessageReviewCommentCard({ comment }: { comment: ReviewCommentContext }) {
  const ctx = use(TimelineRowCtx);
  const fenceLanguage = comment.fenceLanguage ?? "diff";
  const renderablePatch = getRenderablePatch(
    buildReviewCommentRenderablePatch(comment),
    `review-comment:${comment.id}`,
  );

  return (
    <div className="space-y-2 rounded-lg border border-border/70 bg-background/70 p-3">
      <div className="space-y-1">
        <div className="text-xs font-medium text-foreground">
          {formatWorkspaceRelativePath(comment.filePath, ctx.workspaceRoot)}
        </div>
        <div className="text-[11px] text-muted-foreground">
          {comment.sectionTitle} · {comment.rangeLabel}
        </div>
      </div>
      {comment.text.length > 0 && (
        <div className="whitespace-pre-wrap wrap-break-word text-sm">
          <SkillInlineText text={comment.text} skills={ctx.skills} />
        </div>
      )}
      {fenceLanguage !== "diff" && comment.diff.trim().length > 0 && (
        <ChatMarkdown
          text={formatReviewCommentFence(fenceLanguage, comment.diff)}
          cwd={ctx.markdownCwd}
          threadRef={ctx.threadRef ?? undefined}
          skills={ctx.skills}
          className="text-foreground"
        />
      )}
      {renderablePatch?.kind === "files" &&
        renderablePatch.files.map((fileDiff) => (
          <FileDiff
            key={resolveFileDiffPath(fileDiff)}
            fileDiff={fileDiff}
            options={{
              collapsed: false,
              diffStyle: "unified",
              theme: resolveDiffThemeName(ctx.resolvedTheme),
            }}
          />
        ))}
      {renderablePatch?.kind === "raw" && (
        <pre className="overflow-x-auto rounded-md bg-muted/40 p-2 text-xs">
          {renderablePatch.text}
        </pre>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Structural sharing — reuse old row references when data hasn't changed
// so LegendList (and React) can skip re-rendering unchanged items.
// ---------------------------------------------------------------------------

/** Returns a structurally-shared copy of `rows`: for each row whose content
 *  hasn't changed since last call, the previous object reference is reused. */
function useStableRows(rows: MessagesTimelineRow[]): MessagesTimelineRow[] {
  const prevState = useRef<StableMessagesTimelineRowsState>({
    byId: new Map<string, MessagesTimelineRow>(),
    result: [],
  });

  return useMemo(() => {
    const nextState = computeStableMessagesTimelineRows(rows, prevState.current);
    prevState.current = nextState;
    return nextState.result;
  }, [rows]);
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

function formatWorkingTimer(startIso: string, endIso: string): string | null {
  const startedAtMs = Date.parse(startIso);
  const endedAtMs = Date.parse(endIso);
  if (!Number.isFinite(startedAtMs) || !Number.isFinite(endedAtMs)) {
    return null;
  }

  const elapsedSeconds = Math.max(0, Math.floor((endedAtMs - startedAtMs) / 1000));
  if (elapsedSeconds < 60) {
    return `${elapsedSeconds}s`;
  }

  const hours = Math.floor(elapsedSeconds / 3600);
  const minutes = Math.floor((elapsedSeconds % 3600) / 60);
  const seconds = elapsedSeconds % 60;

  if (hours > 0) {
    return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
  }

  return seconds > 0 ? `${minutes}m ${seconds}s` : `${minutes}m`;
}

function formatWorkingTimerNow(startIso: string): string {
  return formatWorkingTimer(startIso, new Date().toISOString()) ?? "0s";
}

type WorkEntryIconName =
  | "bot"
  | "check"
  | "circle-alert"
  | "eye"
  | "globe"
  | "hammer"
  | "message-circle"
  | "square-pen"
  | "terminal"
  | "wrench"
  | "x"
  | "zap";

function WorkEntryIconSvg({ name, className }: { name: WorkEntryIconName; className: string }) {
  switch (name) {
    case "bot":
      return <BotIcon className={className} aria-hidden />;
    case "check":
      return <CheckIcon className={className} aria-hidden />;
    case "circle-alert":
      return <CircleAlertIcon className={className} aria-hidden />;
    case "eye":
      return <EyeIcon className={className} aria-hidden />;
    case "globe":
      return <GlobeIcon className={className} aria-hidden />;
    case "hammer":
      return <HammerIcon className={className} aria-hidden />;
    case "message-circle":
      return <MessageCircleIcon className={className} aria-hidden />;
    case "square-pen":
      return <SquarePenIcon className={className} aria-hidden />;
    case "terminal":
      return <TerminalIcon className={className} aria-hidden />;
    case "wrench":
      return <WrenchIcon className={className} aria-hidden />;
    case "x":
      return <XIcon className={className} aria-hidden />;
    case "zap":
      return <ZapIcon className={className} aria-hidden />;
  }
}

function workToneIcon(tone: TimelineWorkEntry["tone"]): {
  iconName: WorkEntryIconName;
  className: string;
} {
  if (tone === "error") {
    return {
      iconName: "circle-alert",
      className: "text-foreground/92",
    };
  }
  if (tone === "thinking") {
    return {
      iconName: "bot",
      className: "text-foreground/92",
    };
  }
  if (tone === "info") {
    return {
      iconName: "check",
      className: "text-muted-foreground",
    };
  }
  return {
    iconName: "zap",
    className: "text-foreground/92",
  };
}

function workEntryPreview(
  workEntry: Pick<TimelineWorkEntry, "detail" | "command" | "changedFiles">,
  workspaceRoot: string | undefined,
) {
  if (workEntry.command) return workEntry.command;
  if (workEntry.detail) return workEntry.detail;
  if ((workEntry.changedFiles?.length ?? 0) === 0) return null;
  const [firstPath] = workEntry.changedFiles ?? [];
  if (!firstPath) return null;
  const displayPath = formatWorkspaceRelativePath(firstPath, workspaceRoot);
  return workEntry.changedFiles!.length === 1
    ? displayPath
    : `${displayPath} +${workEntry.changedFiles!.length - 1} more`;
}

function workEntryRawCommand(
  workEntry: Pick<TimelineWorkEntry, "command" | "rawCommand">,
): string | null {
  const rawCommand = workEntry.rawCommand?.trim();
  if (!rawCommand || !workEntry.command) {
    return null;
  }
  return rawCommand === workEntry.command.trim() ? null : rawCommand;
}

function buildToolCallExpandedBody(
  workEntry: TimelineWorkEntry,
  workspaceRoot: string | undefined,
): string | null {
  const blocks: string[] = [];
  if (workEntry.itemType === "mcp_tool_call" && workEntry.toolData !== undefined) {
    blocks.push(`MCP call\n${JSON.stringify(workEntry.toolData, null, 2)}`);
  }
  const raw = workEntryRawCommand(workEntry);
  if (raw?.trim()) {
    blocks.push(raw.trim());
  } else if (workEntry.command?.trim()) {
    blocks.push(workEntry.command.trim());
  }
  if (workEntry.detail?.trim()) {
    blocks.push(workEntry.detail.trim());
  }
  const changedFiles = workEntry.changedFiles ?? [];
  if (changedFiles.length > 0) {
    blocks.push(
      changedFiles
        .map((filePath) => formatWorkspaceRelativePath(filePath, workspaceRoot))
        .join("\n"),
    );
  }
  const metadata = [
    workEntry.cwd ? `cwd: ${workEntry.cwd}` : null,
    workEntry.durationMs !== undefined ? `duration: ${formatDuration(workEntry.durationMs)}` : null,
    workEntry.exitCode !== undefined ? `exit code: ${workEntry.exitCode}` : null,
  ].filter((value): value is string => value !== null);
  if (metadata.length > 0) blocks.unshift(metadata.join(" · "));
  return blocks.length > 0 ? blocks.join("\n\n") : null;
}

function workEntryStatusLabel(workEntry: TimelineWorkEntry): string | null {
  if (workEntry.sourceActivityKind === "user-input.requested") return "Waiting";
  switch (workEntry.toolLifecycleStatus) {
    case "inProgress":
      return "Running";
    case "completed":
      return "Succeeded";
    case "failed":
      return "Failed";
    case "declined":
    case "stopped":
      return "Canceled";
    default:
      return null;
  }
}

function workEntryIconName(workEntry: TimelineWorkEntry): WorkEntryIconName {
  if (
    workEntry.sourceActivityKind === "user-input.requested" ||
    workEntry.sourceActivityKind === "user-input.resolved"
  ) {
    return "message-circle";
  }
  if (workEntry.requestKind === "command") return "terminal";
  if (workEntry.requestKind === "file-read") return "eye";
  if (workEntry.requestKind === "file-change") return "square-pen";

  if (workEntry.itemType === "command_execution" || workEntry.command) {
    return "terminal";
  }
  if (workEntry.itemType === "file_change" || (workEntry.changedFiles?.length ?? 0) > 0) {
    return "square-pen";
  }
  if (workEntry.itemType === "web_search") return "globe";
  if (workEntry.itemType === "image_view") return "eye";

  switch (workEntry.itemType) {
    case "mcp_tool_call":
      return "wrench";
    case "dynamic_tool_call":
    case "collab_agent_tool_call":
      return "hammer";
  }

  return workToneIcon(workEntry.tone).iconName;
}

function capitalizePhrase(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    return value;
  }
  return `${trimmed.charAt(0).toUpperCase()}${trimmed.slice(1)}`;
}

function toolWorkEntryHeading(workEntry: TimelineWorkEntry): string {
  if (!workEntry.toolTitle) {
    return capitalizePhrase(normalizeCompactToolLabel(workEntry.label));
  }
  return capitalizePhrase(normalizeCompactToolLabel(workEntry.toolTitle));
}

const stopRowToggle = (e: { stopPropagation: () => void }) => e.stopPropagation();

const SimpleWorkEntryRow = memo(function SimpleWorkEntryRow(props: {
  workEntry: TimelineWorkEntry;
  workspaceRoot: string | undefined;
}) {
  const { workEntry, workspaceRoot } = props;
  const activity = use(TimelineRowActivityCtx);
  const timeline = use(TimelineRowCtx);
  const [expanded, setExpanded] = useState(false);
  const iconConfig = workToneIcon(workEntry.tone);
  const showWarningIndicator = workEntry.sourceActivityKind === "runtime.warning";
  const entryIconName = showWarningIndicator ? "x" : workEntryIconName(workEntry);
  const heading = toolWorkEntryHeading(workEntry);
  const rawPreview = workEntryPreview(workEntry, workspaceRoot);
  const preview =
    rawPreview &&
    normalizeCompactToolLabel(rawPreview).toLowerCase() ===
      normalizeCompactToolLabel(heading).toLowerCase()
      ? null
      : rawPreview;
  const displayText = preview ? `${heading} - ${preview}` : heading;
  const statusLabel = workEntryStatusLabel(workEntry);
  const compactMetadata = [
    workEntry.durationMs !== undefined ? formatDuration(workEntry.durationMs) : null,
    workEntry.exitCode !== undefined ? `exit ${workEntry.exitCode}` : null,
  ].filter((value): value is string => value !== null);
  const expandedBody = buildToolCallExpandedBody(workEntry, workspaceRoot);
  const canExpand = expandedBody !== null;
  const showFailedIndicator = workEntryIndicatesToolFailure(workEntry);
  const showDestructiveRowStyle =
    showFailedIndicator &&
    (workEntry.sourceActivityKind === "runtime.error" || !workLogEntryIsToolLike(workEntry));
  const iconWrapperClass = cn(
    "flex size-5 shrink-0 items-center justify-center",
    showWarningIndicator
      ? "text-destructive"
      : showDestructiveRowStyle
        ? "text-destructive"
        : workEntry.tone === "tool" || showFailedIndicator
          ? "text-muted-foreground/65"
          : iconConfig.className,
  );
  const headingClass = showWarningIndicator
    ? "font-medium text-warning"
    : showDestructiveRowStyle
      ? "font-medium text-destructive"
      : "font-medium text-foreground/82";
  const turnSettled = !activity.activeTurnInProgress;
  const showNeutralIndicator = !turnSettled && workEntryIndicatesToolNeutralStatus(workEntry);
  const showSuccessIndicator =
    workEntryIndicatesToolSuccess(workEntry) ||
    (turnSettled && workEntryIndicatesToolNeutralStatus(workEntry));
  const toggleExpanded = () => {
    timeline.onFoldAnimationStart();
    setExpanded((value) => !value);
  };
  const rowToggleProps = canExpand
    ? {
        role: "button" as const,
        tabIndex: 0 as const,
        "aria-label": displayText,
        "aria-expanded": expanded,
        onClick: toggleExpanded,
        onKeyDown: (e: KeyboardEvent<HTMLDivElement>) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            toggleExpanded();
          }
        },
      }
    : {};

  return (
    <Collapsible open={expanded}>
      <div
        className={cn(
          "group flex flex-col rounded-md px-0.5 py-px transition-colors",
          canExpand &&
            "cursor-pointer hover:bg-accent/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/70",
        )}
        {...rowToggleProps}
      >
        <div className="flex select-none items-center gap-1.5 transition-[opacity,translate] duration-200">
          <span className={iconWrapperClass}>
            <WorkEntryIconSvg
              name={entryIconName}
              className="block size-3.5 shrink-0 stroke-[1.8] opacity-80"
            />
          </span>
          <div className="flex min-w-0 flex-1 items-center gap-1.5">
            <div className="min-w-0 flex-1 overflow-hidden">
              <p className="flex min-w-0 w-full items-baseline gap-1.5 text-[12px] leading-5">
                <span className={cn("min-w-0 shrink truncate", headingClass)}>{heading}</span>
                {preview && (
                  <span className="min-w-0 flex-1 truncate text-muted-foreground/55">
                    {preview}
                  </span>
                )}
              </p>
            </div>
            <div
              className={cn(
                "flex shrink-0 items-center gap-px text-muted-foreground/55 transition-opacity",
                showFailedIndicator || statusLabel === "Running"
                  ? "opacity-100"
                  : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100",
              )}
            >
              {compactMetadata.length > 0 ? (
                <span className="me-1 text-[10px] tabular-nums">{compactMetadata.join(" · ")}</span>
              ) : null}
              {statusLabel ? (
                <span className="me-1 text-[10px] font-medium uppercase tracking-wide">
                  {statusLabel}
                </span>
              ) : null}
              {expandedBody ? (
                <span onClick={stopRowToggle} onPointerDown={stopRowToggle}>
                  <MessageCopyButton
                    text={expandedBody}
                    size="icon-xs"
                    variant="ghost"
                    className="size-4"
                  />
                </span>
              ) : null}
              <span
                className="flex size-4 shrink-0 items-center justify-center"
                aria-hidden={!canExpand}
              >
                {canExpand ? (
                  <ChevronDownIcon
                    className={cn(
                      "size-3 shrink-0 opacity-70 transition-transform duration-200",
                      expanded && "rotate-180",
                    )}
                    aria-hidden
                  />
                ) : null}
              </span>
              <span className="flex size-4 shrink-0 items-center justify-center">
                {showFailedIndicator ? (
                  <Tooltip>
                    <TooltipTrigger
                      render={
                        <span
                          className="flex size-4 items-center justify-center"
                          aria-label="Tool call failed"
                        />
                      }
                    >
                      <XIcon className="block size-3 shrink-0 text-destructive" aria-hidden />
                    </TooltipTrigger>
                    <TooltipPopup>Failed</TooltipPopup>
                  </Tooltip>
                ) : showSuccessIndicator ? (
                  <Tooltip>
                    <TooltipTrigger
                      render={<span className="flex size-4 items-center justify-center" />}
                    >
                      <span className="inline-flex size-4 items-center justify-center">
                        <CheckIcon
                          className="block size-3 shrink-0 stroke-current"
                          stroke="currentColor"
                          aria-hidden
                        />
                      </span>
                    </TooltipTrigger>
                    <TooltipPopup>Completed</TooltipPopup>
                  </Tooltip>
                ) : showNeutralIndicator ? (
                  <Tooltip>
                    <TooltipTrigger
                      render={<span className="flex size-4 items-center justify-center" />}
                    >
                      <MinusIcon className="block size-3 shrink-0 opacity-70" aria-hidden />
                    </TooltipTrigger>
                    <TooltipPopup>Empty</TooltipPopup>
                  </Tooltip>
                ) : null}
              </span>
            </div>
          </div>
        </div>
        {canExpand && expandedBody ? (
          <CollapsiblePanel>
            <div
              className="mt-1 ms-7 cursor-default border-s border-border/45 ps-3 pb-2 pt-0.5"
              onClick={stopRowToggle}
              onPointerDown={stopRowToggle}
            >
              <pre className="max-h-64 cursor-text overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-muted-foreground select-text">
                {expandedBody}
              </pre>
            </div>
          </CollapsiblePanel>
        ) : null}
      </div>
    </Collapsible>
  );
});
