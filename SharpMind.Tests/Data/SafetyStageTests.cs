using SharpMind.Data.Pipeline.Stages;

namespace SharpMind.Tests.Data;

public sealed class SafetyStageTests
{
    // BlocklistFilter

    [Fact]
    public void Blocklist_DiscardsDocumentContainingTerm()
    {
        var stage = new BlocklistFilter(["badword"]);
        Assert.Null(stage.Process("this contains badword in it"));
    }

    [Fact]
    public void Blocklist_PassesCleanDocument()
    {
        var stage = new BlocklistFilter(["badword"]);
        Assert.Equal("this is fine", stage.Process("this is fine"));
    }

    [Fact]
    public void Blocklist_CaseInsensitive()
    {
        var stage = new BlocklistFilter(["badword"]);
        Assert.Null(stage.Process("BADWORD appears here"));
    }

    [Fact]
    public void Blacklist_WholeWord_DoesNotMatchPartial()
    {
        var stage = new BlacklistFilter(["ass"]);
        Assert.NotNull(stage.Process("assignment is due"));  // "ass" inside word
        Assert.Null(stage.Process("what an ass he is"));     // standalone
    }

    [Fact]
    public void Blocklist_NonWholeWord_MatchesPartial()
    {
        var stage = new BlocklistFilter(["ass"]);
        Assert.Null(stage.Process("assignment is due")); // matches anywhere
    }

    [Fact]
    public void Blocklist_MultipleTerms_AnyMatchDiscards()
    {
        var stage = new BlocklistFilter(["foo", "bar"]);
        Assert.Null(stage.Process("contains foo"));
        Assert.Null(stage.Process("contains bar"));
        Assert.NotNull(stage.Process("contains neither"));
    }

    [Fact]
    public void Blocklist_FromFile_LoadsTerms()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "# comment\nbadterm\n  another  \n");
            var stage = new BlocklistFilter(path);
            Assert.Null(stage.Process("this has badterm in it"));
            Assert.Null(stage.Process("this has another one"));
            Assert.NotNull(stage.Process("this is clean"));
        }
        finally { File.Delete(path); }
    }

    // HapFilter

    [Fact]
    public void Hap_Discard_RemovesDocumentWithHapContent()
    {
        var stage = new HapFilter(HapFilter.HapMode.Discard, []);
        Assert.Null(stage.Process($"this is {HapFilter.Decrypt(["gvdl"])[0]} terrible"));
    }

    [Fact]
    public void Hap_Discard_PassesCleanDocument()
    {
        var stage = new HapFilter(HapFilter.HapMode.Discard, []);
        Assert.Equal("this is fine", stage.Process("this is fine"));
    }

    [Fact]
    public void Hap_Redact_ReplacesMatchWithPlaceholder()
    {
        var badword = HapFilter.Decrypt(["gvdl"])[0];
        var stage  = new HapFilter(HapFilter.HapMode.Redact, []);
        string? r  = stage.Process($"what the {badword} is this");
        Assert.NotNull(r);
        Assert.Contains("[HAP]", r);
        Assert.DoesNotContain(badword, r);
    }

    [Fact]
    public void Hap_Redact_KeepsRestOfDocument()
    {
        var badword = HapFilter.Decrypt(["gvdl"])[0];
        var stage  = new HapFilter(HapFilter.HapMode.Redact, []);
        string? r  = stage.Process($"hello {badword} world");
        Assert.NotNull(r);
        Assert.Contains("hello", r);
        Assert.Contains("world", r);
    }

    [Fact]
    public void Hap_AdditionalTerms_AlsoBlocked()
    {
        var stage = new HapFilter(HapFilter.HapMode.Discard, ["newbadterm"]);
        Assert.Null(stage.Process("this has newbadterm in it"));
    }

    // PiiMaskingStage

    [Fact]
    public void Pii_MasksEmail()
    {
        var stage  = new PiiMasker(PiiType.Email);
        string? r  = stage.Process("contact me at user@example.com please");
        Assert.NotNull(r);
        Assert.Contains("[EMAIL]", r);
        Assert.DoesNotContain("user@example.com", r);
    }

    [Fact]
    public void Pii_MasksUsPhone()
    {
        var stage  = new PiiMasker(PiiType.Phone);
        string? r  = stage.Process("call me at 555-867-5309");
        Assert.NotNull(r);
        Assert.Contains("[PHONE]", r);
    }

    [Fact]
    public void Pii_MasksIpv4()
    {
        var stage  = new PiiMasker(PiiType.IpAddress);
        string? r  = stage.Process("server at 192.168.1.1 is down");
        Assert.NotNull(r);
        Assert.Contains("[IP_ADDRESS]", r);
        Assert.DoesNotContain("192.168.1.1", r);
    }

    [Fact]
    public void Pii_MasksSsn()
    {
        var stage  = new PiiMasker(PiiType.Ssn);
        string? r  = stage.Process("SSN is 123-45-6789");
        Assert.NotNull(r);
        Assert.Contains("[SSN]", r);
        Assert.DoesNotContain("123-45-6789", r);
    }

    [Fact]
    public void Pii_MasksUrl()
    {
        var stage  = new PiiMasker(PiiType.Url);
        string? r  = stage.Process("see https://example.com/page for details");
        Assert.NotNull(r);
        Assert.Contains("[URL]", r);
    }

    [Fact]
    public void Pii_SelectiveTypes_OnlyMasksEnabled()
    {
        var stage  = new PiiMasker(PiiType.Email);
        string? r  = stage.Process("email: user@example.com phone: 555-867-5309");
        Assert.NotNull(r);
        Assert.Contains("[EMAIL]", r);
        Assert.DoesNotContain("[PHONE]", r);  // phone not enabled
        Assert.Contains("555-867-5309", r);
    }

    [Fact]
    public void Pii_NoMatch_ReturnsDocumentUnchanged()
    {
        var stage = new PiiMasker(PiiType.All);
        string doc = "this document has no PII whatsoever";
        Assert.Equal(doc, stage.Process(doc));
    }

    [Fact]
    public void Pii_AllTypes_MasksEverything()
    {
        var stage  = new PiiMasker(PiiType.All);
        string doc = "email user@x.com, ip 1.2.3.4, ssn 123-45-6789, url https://x.com";
        string? r  = stage.Process(doc);
        Assert.NotNull(r);
        Assert.DoesNotContain("user@x.com",   r);
        Assert.DoesNotContain("1.2.3.4",      r);
        Assert.DoesNotContain("123-45-6789",  r);
        Assert.DoesNotContain("https://x.com", r);
    }
}
