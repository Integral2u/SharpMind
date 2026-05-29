using System.Text;
using System.Text.RegularExpressions;
using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat.PromptFormatters;

/// <summary>
/// Executes the Jinja2 subset used by GGUF <c>tokenizer.chat_template</c> strings.
///
/// Supported constructs (covering every template seen in practice):
///   {%- for message in messages %} … {% endfor %}
///   {%- if expr %} … {%- elif expr %} … {%- else %} … {%- endif %}
///   {%- set var = expr %}
///   {%- set ns = namespace(field=value, …) %}  and  ns.field  assignments
///   {{ expr }}          — output expression
///   loop.first / loop.last / loop.index0
///   message['role'] / message.role / message['content'] / message.content
///   messages[0]['role'] / messages[loop.index0 ± N]
///   bos_token, eos_token  — resolved from the Tokenizer at render time
///   add_generation_prompt — bool passed by the caller (always true for inference)
///   X is none / X is not none / X is defined / not X is defined
///   'str' in X  (substring / key test)
///   X + Y  (string concatenation)
///   content.split('delim')[-1]  (last segment after split)
///   X | trim  (whitespace trim filter)
///   tool_calls / tool call blocks are intentionally skipped (plain-text chat only)
/// </summary>
public sealed class JinjaTemplateFormatter : IChatPromptFormatter
{
    private readonly string _template;

    public JinjaTemplateFormatter(string template)
    {
        _template = template;
    }

    // ── Public entry point ────────────────────────────────────────────────

    public string Format(IReadOnlyList<ChatMessage> history, Tokenizer tokenizer, bool addBos)
    {
        // Build the variable context that templates can reference.
        string bosToken = addBos && tokenizer.BosId >= 0
            ? tokenizer.IdToToken(tokenizer.BosId) : "";
        string eosToken = tokenizer.EosId >= 0
            ? tokenizer.IdToToken(tokenizer.EosId) : "";

        // Convert ChatMessage list to the dict-of-strings representation
        // that templates expect: message['role'] and message['content'].
        var messages = history.Select(m => new JinjaDict
        {
            ["role"] = RoleName(m.Role),
            ["content"] = (object)m.Content
        }).ToList();

        var env = new JinjaEnv();
        env.Set("messages", messages);
        env.Set("bos_token", bosToken);
        env.Set("eos_token", eosToken);
        env.Set("add_generation_prompt", (object)true);  // always true for inference

        var tokens = Tokenise(_template);
        var sb = new StringBuilder();
        Execute(tokens, 0, tokens.Count, env, sb);
        return sb.ToString();
    }

    // ── Tokeniser ─────────────────────────────────────────────────────────
    // Splits the template into a flat list of Token objects.

    private enum TKind { Text, Output, Tag }

    private sealed record Token(TKind Kind, string Body, bool StripLeft, bool StripRight);

    private static List<Token> Tokenise(string src)
    {
        var result = new List<Token>();
        int pos = 0;
        while (pos < src.Length)
        {
            int tag = src.IndexOf("{%", pos, StringComparison.Ordinal);
            int expr = src.IndexOf("{{", pos, StringComparison.Ordinal);

            // Find whichever opener comes first.
            int next = (tag < 0 && expr < 0) ? -1
                      : tag < 0 ? expr
                      : expr < 0 ? tag
                      : Math.Min(tag, expr);

            if (next < 0)
            {
                if (pos < src.Length) result.Add(new Token(TKind.Text, src[pos..], false, false));
                break;
            }

            if (next > pos)
                result.Add(new Token(TKind.Text, src[pos..next], false, false));

            if (next == expr && (tag < 0 || expr < tag))
            {
                int end = src.IndexOf("}}", next + 2, StringComparison.Ordinal);
                if (end < 0) { result.Add(new Token(TKind.Text, src[next..], false, false)); break; }
                string body = src[(next + 2)..(end)];
                result.Add(new Token(TKind.Output, body.Trim(), false, false));
                pos = end + 2;
            }
            else
            {
                int end = src.IndexOf("%}", next + 2, StringComparison.Ordinal);
                if (end < 0) { result.Add(new Token(TKind.Text, src[next..], false, false)); break; }
                string raw = src[(next + 2)..end];
                bool sl = raw.StartsWith('-');
                bool sr = raw.EndsWith('-');
                string body = raw.Trim().TrimStart('-').TrimEnd('-').Trim();
                result.Add(new Token(TKind.Tag, body, sl, sr));
                pos = end + 2;
                // strip-right: eat the immediately following newline
                if (sr && pos < src.Length && src[pos] == '\n') pos++;
            }
        }
        return result;
    }

    // ── Executor ──────────────────────────────────────────────────────────

    private static void Execute(
        List<Token> tokens, int start, int end,
        JinjaEnv env, StringBuilder sb)
    {
        int i = start;
        while (i < end)
        {
            var tok = tokens[i];

            if (tok.Kind == TKind.Text)
            {
                sb.Append(tok.Body);
                i++;
                continue;
            }

            if (tok.Kind == TKind.Output)
            {
                var val = Eval(tok.Body, env);
                if (val != null) sb.Append(Stringify(val));
                i++;
                continue;
            }

            // Tag
            string tag = tok.Body;

            if (tag.StartsWith("for ", StringComparison.Ordinal))
            {
                i = ExecuteFor(tokens, i, end, env, sb);
                continue;
            }

            if (tag.StartsWith("if ", StringComparison.Ordinal) || tag == "if")
            {
                i = ExecuteIf(tokens, i, end, env, sb);
                continue;
            }

            if (tag.StartsWith("set ", StringComparison.Ordinal))
            {
                ExecuteSet(tag, env);
                i++;
                continue;
            }

            // endfor / endif / else / elif — handled by their parent, skip
            i++;
        }
    }

    // ── For loop ──────────────────────────────────────────────────────────

    private static int ExecuteFor(
        List<Token> tokens, int forIdx, int end,
        JinjaEnv env, StringBuilder sb)
    {
        // Parse "for VAR in EXPR"
        var m = Regex.Match(tokens[forIdx].Body, @"^for\s+(\w+)\s+in\s+(.+)$");
        if (!m.Success) return forIdx + 1;

        string varName = m.Groups[1].Value;
        string iterExpr = m.Groups[2].Value.Trim();
        var iterable = Eval(iterExpr, env);

        // Find matching endfor
        int bodyStart = forIdx + 1;
        int bodyEnd = FindMatchingEnd(tokens, forIdx, end, "for", "endfor");

        if (iterable is List<JinjaDict> list)
        {
            int count = list.Count;
            for (int idx = 0; idx < count; idx++)
            {
                var child = env.Push();
                child.Set(varName, list[idx]);

                // loop metadata object
                var loop = new JinjaDict
                {
                    ["first"] = (object)(idx == 0),
                    ["last"] = (object)(idx == count - 1),
                    ["index0"] = (object)(long)idx,
                    ["index"] = (object)(long)(idx + 1)
                };
                child.Set("loop", loop);

                Execute(tokens, bodyStart, bodyEnd, child, sb);
            }
        }

        return bodyEnd + 1; // +1 to skip the endfor token
    }

    // ── If / elif / else ──────────────────────────────────────────────────

    private static int ExecuteIf(
        List<Token> tokens, int ifIdx, int end,
        JinjaEnv env, StringBuilder sb)
    {
        // Collect all branches: [(condition_expr, start, end), …]
        // null condition = else branch
        var branches = new List<(string? Cond, int Start, int End)>();
        int depth = 0;
        int branchStart = ifIdx + 1;
        string? curCond = tokens[ifIdx].Body["if ".Length..].Trim();

        for (int i = ifIdx + 1; i < end; i++)
        {
            if (tokens[i].Kind != TKind.Tag) continue;
            string t = tokens[i].Body;

            if (t.StartsWith("for ", StringComparison.Ordinal) ||
                t.StartsWith("if ", StringComparison.Ordinal)) { depth++; continue; }

            if ((t == "endfor" || t == "endif") && depth > 0) { depth--; continue; }

            if (depth > 0) continue;

            if (t == "endif")
            {
                branches.Add((curCond, branchStart, i));
                // Execute first branch whose condition is true
                foreach (var (cond, bs, be) in branches)
                {
                    if (cond == null || IsTruthy(Eval(cond, env)))
                    {
                        Execute(tokens, bs, be, env, sb);
                        break;
                    }
                }
                return i + 1;
            }

            if (t.StartsWith("elif ", StringComparison.Ordinal) ||
                t.StartsWith("else if ", StringComparison.Ordinal))
            {
                branches.Add((curCond, branchStart, i));
                curCond = t.StartsWith("elif ") ? t["elif ".Length..].Trim()
                                                      : t["else if ".Length..].Trim();
                branchStart = i + 1;
                continue;
            }

            if (t == "else")
            {
                branches.Add((curCond, branchStart, i));
                curCond = null;
                branchStart = i + 1;
            }
        }

        // Fell off the end without finding endif — execute what we have
        branches.Add((curCond, branchStart, end));
        foreach (var (cond, bs, be) in branches)
        {
            if (cond == null || IsTruthy(Eval(cond, env)))
            {
                Execute(tokens, bs, be, env, sb);
                break;
            }
        }
        return end;
    }

    // ── Set ───────────────────────────────────────────────────────────────

    private static void ExecuteSet(string tag, JinjaEnv env)
    {
        // set ns = namespace(field=val, …)
        var nsMatch = Regex.Match(tag,
            @"^set\s+(\w+)\s*=\s*namespace\s*\((.*)?\)\s*$",
            RegexOptions.Singleline);
        if (nsMatch.Success)
        {
            string name = nsMatch.Groups[1].Value;
            string args = nsMatch.Groups[2].Value;
            var ns = new JinjaNamespace();
            foreach (Match kv in Regex.Matches(args, @"(\w+)\s*=\s*([^,]+)"))
            {
                string k = kv.Groups[1].Value.Trim();
                string v = kv.Groups[2].Value.Trim();
                ns.Set(k, Eval(v, env) ?? (object)"");
            }
            env.Set(name, ns);
            return;
        }

        // set ns.field = expr
        var dotMatch = Regex.Match(tag, @"^set\s+(\w+)\.(\w+)\s*=\s*(.+)$");
        if (dotMatch.Success)
        {
            string obj = dotMatch.Groups[1].Value;
            string field = dotMatch.Groups[2].Value;
            string expr = dotMatch.Groups[3].Value.Trim();
            var target = env.Get(obj);
            if (target is JinjaNamespace jns)
                jns.Set(field, Eval(expr, env) ?? (object)"");
            else if (target is JinjaDict jd)
                jd[field] = Eval(expr, env) ?? (object)"";
            return;
        }

        // set var = expr
        var simple = Regex.Match(tag, @"^set\s+(\w+)\s*=\s*(.+)$");
        if (simple.Success)
        {
            string name = simple.Groups[1].Value;
            string expr = simple.Groups[2].Value.Trim();
            env.Set(name, Eval(expr, env) ?? (object)"");
        }
    }

    // ── Expression evaluator ──────────────────────────────────────────────

    public static object? Eval(string expr, JinjaEnv env)
    {
        expr = expr.Trim();

        // strip outer parens
        if (expr.StartsWith('(') && expr.EndsWith(')'))
            return Eval(expr[1..^1], env);

        // 'not X is defined'
        if (Regex.IsMatch(expr, @"^not\s+\w+\s+is\s+defined$"))
        {
            string vname = Regex.Match(expr, @"not\s+(\w+)").Groups[1].Value;
            return (object)(env.Get(vname) == null);
        }

        // X is defined
        if (Regex.IsMatch(expr, @"^(\w+)\s+is\s+defined$"))
        {
            string vname = Regex.Match(expr, @"^(\w+)").Groups[1].Value;
            return (object)(env.Get(vname) != null);
        }

        // X is none / X is not none
        var noneM = Regex.Match(expr, @"^(.+?)\s+is\s+(not\s+)?none$");
        if (noneM.Success)
        {
            var v = Eval(noneM.Groups[1].Value, env);
            bool isNone = v == null || v is string s && s == "";
            return (object)(noneM.Groups[2].Success ? !isNone : isNone);
        }

        // not X
        if (expr.StartsWith("not ", StringComparison.Ordinal))
            return (object)!IsTruthy(Eval(expr[4..], env));

        // 'literal' in X  (substring test)
        var inMatch = Regex.Match(expr, @"^'([^']*)'\s+in\s+(.+)$");
        if (inMatch.Success)
        {
            string needle = inMatch.Groups[1].Value;
            var hay = Eval(inMatch.Groups[2].Value, env);
            string hayStr = hay is string hs ? hs : Stringify(hay);
            return (object)hayStr.Contains(needle, StringComparison.Ordinal);
        }

        // "literal" in X
        var inMatch2 = Regex.Match(expr, "^\"([^\"]*)\"\\s+in\\s+(.+)$");
        if (inMatch2.Success)
        {
            string needle = inMatch2.Groups[1].Value;
            var hay = Eval(inMatch2.Groups[2].Value, env);
            string hayStr = hay is string hs ? hs : Stringify(hay);
            return (object)hayStr.Contains(needle, StringComparison.Ordinal);
        }

        // X == Y
        var eqM = Regex.Match(expr, @"^(.+?)\s*==\s*(.+)$");
        if (eqM.Success)
        {
            var left = Eval(eqM.Groups[1].Value, env);
            var right = Eval(eqM.Groups[2].Value, env);
            return (object)(Stringify(left) == Stringify(right));
        }

        // X != Y
        var neM = Regex.Match(expr, @"^(.+?)\s*!=\s*(.+)$");
        if (neM.Success)
        {
            var left = Eval(neM.Groups[1].Value, env);
            var right = Eval(neM.Groups[2].Value, env);
            return (object)(Stringify(left) != Stringify(right));
        }

        // X and Y
        var andM = Regex.Match(expr, @"^(.+?)\s+and\s+(.+)$");
        if (andM.Success)
            return (object)(IsTruthy(Eval(andM.Groups[1].Value, env)) &&
                            IsTruthy(Eval(andM.Groups[2].Value, env)));

        // X or Y
        var orM = Regex.Match(expr, @"^(.+?)\s+or\s+(.+)$");
        if (orM.Success)
        {
            var lv = Eval(orM.Groups[1].Value, env);
            return IsTruthy(lv) ? lv : Eval(orM.Groups[2].Value, env);
        }

        // String concatenation: expr + expr
        // Must avoid splitting string literals — find + outside quotes
        int plusIdx = FindOperatorOutsideQuotes(expr, '+');
        if (plusIdx >= 0)
        {
            var left = Eval(expr[..plusIdx], env);
            var right = Eval(expr[(plusIdx + 1)..], env);
            return (object)(Stringify(left) + Stringify(right));
        }

        // Filter: expr | trim  (only trim supported)
        var filterM = Regex.Match(expr, @"^(.+?)\s*\|\s*trim\s*$");
        if (filterM.Success)
        {
            var v = Eval(filterM.Groups[1].Value, env);
            return (object)(Stringify(v).Trim());
        }

        // content.split('delim')[-1]  — last segment
        var splitM = Regex.Match(expr,
            @"^(\w+)\.split\('([^']*)'\)\[(-?\d+)\]$");
        if (splitM.Success)
        {
            var src = Stringify(Eval(splitM.Groups[1].Value, env));
            string sep = splitM.Groups[2].Value;
            int idx = int.Parse(splitM.Groups[3].Value);
            var parts = src.Split(sep);
            int real = idx < 0 ? parts.Length + idx : idx;
            return (object)(real >= 0 && real < parts.Length ? parts[real] : "");
        }

        // String literal (single or double quotes)
        if ((expr.StartsWith('\'') && expr.EndsWith('\'')) ||
            (expr.StartsWith('"') && expr.EndsWith('"')))
        {
            string lit = expr[1..^1];
            // Unescape \\n -> newline etc.
            lit = lit.Replace("\\n", "\n").Replace("\\t", "\t");
            return (object)lit;
        }

        // Boolean literals
        if (expr == "true") return (object)true;
        if (expr == "false") return (object)false;
        if (expr == "none" || expr == "None") return null;

        // Integer literal
        if (long.TryParse(expr, out long lval)) return (object)lval;

        // obj[key] or obj[expr]
        var idxM = Regex.Match(expr, @"^(.+?)\[(.+)\]$");
        if (idxM.Success)
        {
            var obj = Eval(idxM.Groups[1].Value, env);
            var key = Eval(idxM.Groups[2].Value, env);
            return AccessIndex(obj, key);
        }

        // obj.field  (dot access, handles loop.first, ns.field etc.)
        var dotM = Regex.Match(expr, @"^(\w+)\.(\w+)$");
        if (dotM.Success)
        {
            var obj = env.Get(dotM.Groups[1].Value);
            string fld = dotM.Groups[2].Value;
            return AccessField(obj, fld);
        }

        // arithmetic: index0 +/- N inside bracket is handled by EvalIndex
        var arithM = Regex.Match(expr, @"^(.+?)\s*([+\-])\s*(\d+)$");
        if (arithM.Success)
        {
            var lv = Eval(arithM.Groups[1].Value, env);
            long rn = long.Parse(arithM.Groups[3].Value);
            long ln = lv is long ll ? ll : lv is int ii ? ii : 0;
            return (object)(arithM.Groups[2].Value == "+" ? ln + rn : ln - rn);
        }

        // Plain identifier
        return env.Get(expr);
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static int FindMatchingEnd(
        List<Token> tokens, int start, int end,
        string openKeyword, string closeKeyword)
    {
        int depth = 0;
        for (int i = start + 1; i < end; i++)
        {
            if (tokens[i].Kind != TKind.Tag) continue;
            string t = tokens[i].Body;
            if (t.StartsWith(openKeyword + " ", StringComparison.Ordinal) ||
                t == openKeyword) depth++;
            else if (t == closeKeyword)
            {
                if (depth == 0) return i;
                depth--;
            }
        }
        return end;
    }

    private static object? AccessIndex(object? obj, object? key)
    {
        if (obj is JinjaDict dict)
        {
            string k = Stringify(key);
            return dict.TryGetValue(k, out var v) ? v : null;
        }
        if (obj is List<JinjaDict> list)
        {
            long idx = key is long l ? l : key is int i ? i : 0;
            int real = idx < 0 ? list.Count + (int)idx : (int)idx;
            return (real >= 0 && real < list.Count) ? list[real] : null;
        }
        return null;
    }

    private static object? AccessField(object? obj, string field)
    {
        if (obj is JinjaDict dict)
            return dict.TryGetValue(field, out var v) ? v : null;
        if (obj is JinjaNamespace ns)
            return ns.Get(field);
        return null;
    }

    private static int FindOperatorOutsideQuotes(string s, char op)
    {
        bool inSingle = false, inDouble = false;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '\'' && !inDouble) inSingle = !inSingle;
            else if (c == '"' && !inSingle) inDouble = !inDouble;
            else if (!inSingle && !inDouble && c == op) return i;
        }
        return -1;
    }

    public static bool IsTruthy(object? v) => v switch
    {
        null => false,
        bool b => b,
        long l => l != 0,
        int i => i != 0,
        string s => s.Length > 0,
        List<JinjaDict> l => l.Count > 0,
        _ => true
    };

    private static string Stringify(object? v) => v switch
    {
        null => "",
        bool b => b ? "True" : "False",
        _ => v.ToString() ?? ""
    };

    private static string RoleName(ChatRole role) => role switch
    {
        ChatRole.System => "system",
        ChatRole.Agent => "assistant",
        ChatRole.User => "user",
        _ => "user"
    };
}

// ── Supporting types ──────────────────────────────────────────────────────────

/// <summary>Dictionary representing a Jinja template object (message, loop, etc.).</summary>
public sealed class JinjaDict : Dictionary<string, object?> { }

/// <summary>
/// Jinja <c>namespace()</c> object — allows cross-scope variable mutation via
/// <c>set ns.field = value</c>.
/// </summary>
public sealed class JinjaNamespace
{
    private readonly Dictionary<string, object?> _fields = new();
    public void Set(string k, object? v) => _fields[k] = v;
    public object? Get(string k) => _fields.TryGetValue(k, out var v) ? v : null;
}

/// <summary>
/// Scoped variable environment. Child scopes see parent variables but cannot
/// pollute the parent (matching Jinja2 for-loop scoping rules).
/// </summary>
public sealed class JinjaEnv
{
    private readonly JinjaEnv? _parent;
    private readonly Dictionary<string, object?> _vars = new();

    public JinjaEnv() { _parent = null; }
    private JinjaEnv(JinjaEnv p) { _parent = p; }

    public void Set(string name, object? value) => _vars[name] = value;

    public object? Get(string name)
    {
        if (_vars.TryGetValue(name, out var v)) return v;
        return _parent?.Get(name);
    }

    /// <summary>Creates a child scope (used for for-loop iterations).</summary>
    public JinjaEnv Push() => new(this);
}
