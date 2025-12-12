# LegacyCodeBench UI Design Options

## Julie Zhuo Design Philosophy Applied

Three UI directions following Julie Zhuo's design principles:
- **Behavior First** — UI exists to enable specific user actions
- **Clarity Over Cleverness** — No decoration without purpose
- **Thoughtful Defaults** — Reduce decisions, provide smart defaults
- **State Completeness** — Zero, loading, empty, error, success all designed
- **Invisible UI** — The best UI disappears

---

## 🌱 Option A: Minimalist + Invisible UI

**File:** `option-a-minimalist.html`

### Characteristics
- Monochrome base palette with single accent color
- Heavy use of whitespace
- Clear typographic hierarchy (DM Sans font)
- Minimal iconography
- Advanced options hidden by default
- Components have calm, neutral presence

### Design Tokens
```css
--ink: #0a0a0a;
--ink-muted: #525252;
--surface: #fafafa;
--accent: #0066ff;
```

### Best For
- Users who want the UI to feel "invisible"
- Tasks that are linear and focused
- Documentation or system tools
- Expert users who know what they're looking for

### Julie Zhuo Principles Applied
✓ Invisible UI — user sees progress, not interface
✓ Content-driven layout — score is the hero
✓ Thoughtful defaults — details hidden, expandable

---

## 🌊 Option B: Structured Dashboard UI

**File:** `option-b-dashboard.html`

### Characteristics
- Left-hand navigation sidebar
- Content panels with clear section headers
- Card-based grid layouts
- Quick actions with thoughtful defaults
- Rich summary cards with context
- Strong, predictable spacing system

### Design Tokens
```css
--bg-app: #f4f4f5;
--bg-surface: #ffffff;
--accent: #6366f1;
--success: #22c55e;
```

### Best For
- Multi-step workflows
- Users who need rapid scanning of multiple metrics
- Systems with admin tools or analytics
- Power users managing multiple evaluations

### Julie Zhuo Principles Applied
✓ Behavior first — clear navigation and actions
✓ State completeness — badges, progress, context
✓ Visual hierarchy — cards organize related info

---

## 🔥 Option C: High-Guidance, Task-Based UI

**File:** `option-c-guided.html`

### Characteristics
- Clear step-by-step flows with progress indicator
- Inline tips, helper text, and microcopy
- Emphasis on next-step clarity
- Strong actionable buttons
- Guidance visible; chrome minimized
- Warm, approachable color palette

### Design Tokens
```css
--bg-page: #fffbf5;
--primary: #ea580c;
--success: #16a34a;
--warning: #ca8a04;
```

### Best For
- Novel workflows (first-time users)
- Onboarding-heavy products
- Users who need reassurance + clarity
- Non-expert users learning the system

### Julie Zhuo Principles Applied
✓ Behavior first — clear "what to do next"
✓ Clarity over cleverness — explicit guidance
✓ Thoughtful defaults — suggestions provided
✓ State completeness — pass/warn/fail indicators

---

## Comparison Matrix

| Aspect | Option A | Option B | Option C |
|--------|----------|----------|----------|
| **Primary User** | Expert | Power User | New User |
| **Cognitive Load** | Low | Medium | Low |
| **Information Density** | Sparse | Dense | Focused |
| **Navigation** | Minimal | Full sidebar | Step-by-step |
| **Guidance Level** | None | Contextual | Explicit |
| **Best Viewport** | Any | Large | Any |
| **Aesthetic** | Monochrome | Professional | Warm |

---

## How to Preview

Open any HTML file directly in a browser:

```bash
# Option A - Minimalist
start web/ui-options/option-a-minimalist.html

# Option B - Dashboard  
start web/ui-options/option-b-dashboard.html

# Option C - Guided
start web/ui-options/option-c-guided.html
```

---

## Recommendation

**For LegacyCodeBench v1.0:**

1. **Public Website** → Option A (Minimalist)
   - Leaderboard page
   - Documentation
   - Clean, professional impression

2. **Evaluation Dashboard** → Option B (Dashboard)
   - For users running multiple evaluations
   - Detailed analytics view
   - Admin interface

3. **First-Run Experience** → Option C (Guided)
   - Onboarding new users
   - Interactive evaluation walkthrough
   - "Getting Started" flow

---

## Implementation Notes

All three options share:
- WCAG AA accessible (contrast, keyboard nav)
- Responsive design
- Semantic HTML
- No external dependencies (CSS-only)
- Dark mode ready (with CSS custom properties)

To implement, copy the styles into your build system and adapt the HTML structure to your templating framework.





