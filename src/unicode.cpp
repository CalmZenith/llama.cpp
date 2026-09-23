#include "unicode.h"
#include "unicode-data.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static std::string unicode_cpts_to_utf8(const std::vector<uint32_t> & cps) {
    std::string result;
    for (size_t i = 0; i < cps.size(); ++i) {
        result.append(unicode_cpt_to_utf8(cps[i]));
    }
    return result;
}

// 根据首字节的“长相”，决定要读几个字节，然后把这些分散的比特拼成一个完整的数字。
uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    // 如果首字节是 0xxxxxxx（最高位是 0），说明是 1 字节的 ASCII 字符，如字符A
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    // 如果首字节是 10xxxxxx（最高位是 1，次高位是 0），说明是无效字符
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    // 如果首字节是 110xxxxx（最高位是 110），说明是 2 字节的字符，如字符é
    if (!(utf8[offset + 0] & 0x20)) {
        // 判断是否越界以及多节字符的二个字节是否合法
        if (offset + 1 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    // 如果首字节是 1110xxxx（最高位是 1110），说明是 3 字节的字符，如字符“你”
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    // 如果首字节是 11110xxx（最高位是 11110），说明是 4 字节的字符，如表情包Emoji
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}

//static std::vector<uint16_t> unicode_cpt_to_utf16(uint32_t cpt) {
//    std::vector<uint16_t> result;
//    if (/* 0x0000 <= cpt && */ cpt <= 0xffff) {
//        result.emplace_back(cpt);
//        return result;
//    }
//    if (0x10000 <= cpt && cpt <= 0x10ffff) {
//        result.emplace_back(0xd800 | ((cpt - 0x10000) >> 10));
//        result.emplace_back(0xdc00 | ((cpt - 0x10000) & 0x03ff));
//        return result;
//    }
//    throw std::invalid_argument("failed to convert codepoint to utf16");
//}

//static std::vector<uint16_t> unicode_cpts_to_utf16(const std::vector<uint32_t> & cps) {
//    std::vector<uint16_t> result;
//    for (size_t i = 0; i < cps.size(); ++i) {
//        auto temp = unicode_cpt_to_utf16(cps[i]);
//        result.insert(result.end(), temp.begin(), temp.end());
//    }
//    return result;
//}

//static uint32_t unicode_cpt_from_utf16(const std::vector<uint16_t> & utf16, size_t & offset) {
//    assert(offset < utf16.size());
//    if (((utf16[0] >> 10) << 10) != 0xd800) {
//        auto result = utf16[offset + 0];
//        offset += 1;
//        return result;
//    }
//
//    if (offset + 1 >= utf16.size() || !((utf16[1] & 0xdc00) == 0xdc00)) {
//        throw std::invalid_argument("invalid character");
//    }
//
//    auto result = 0x10000 + (((utf16[0] & 0x03ff) << 10) | (utf16[1] & 0x03ff));
//    offset += 2;
//    return result;
//}

//static std::vector<uint32_t> unicode_cpts_from_utf16(const std::vector<uint16_t> & utf16) {
//    std::vector<uint32_t> result;
//    size_t offset = 0;
//    while (offset < utf16.size()) {
//        result.push_back(unicode_cpt_from_utf16(utf16, offset));
//    }
//    return result;
//}

// 把 Unicode 规则，打散并扩充成能查表的大数组
// 这个函数的作用就是初始化一个 cpt_flags 数组然后这个数组会包括所有的字符无论是字母数字还是标点等等，
// 然后我们挨个对这个数组的元素遍历，设定好它的类别即是 LETTER 还是 NUMBER 等等
static std::vector<unicode_cpt_flags> unicode_cpt_flags_array() {
    // 初始化一个足够大的数组，默认所有字符都是未定义的
    std::vector<unicode_cpt_flags> cpt_flags(MAX_CODEPOINTS, unicode_cpt_flags::UNDEFINED);

    assert (unicode_ranges_flags.begin()[0].first == 0);
    assert (unicode_ranges_flags.begin()[unicode_ranges_flags.size()-1].first == MAX_CODEPOINTS);
    for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
        const auto range_ini = unicode_ranges_flags.begin()[i-1];  // codepoint_ini, flags
        const auto range_end = unicode_ranges_flags.begin()[i];    // codepoint_end, flags
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    for (auto cpt : unicode_set_whitespace) {
        cpt_flags[cpt].is_whitespace = true;
    }

    for (auto p : unicode_map_lowercase) {
        cpt_flags[p.second].is_lowercase = true;
    }

    for (auto p : unicode_map_uppercase) {
        cpt_flags[p.second].is_uppercase = true;
    }

    for (auto &range : unicode_ranges_nfd) {  // start, last, nfd
        cpt_flags[range.nfd].is_nfd = true;
    }

    return cpt_flags;
}

static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(ch) == map.end()) {
            map[ch] = unicode_cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return map;
}

static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    //  '!' to '~' 直接映射
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    // '¡' to '¬' 直接映射
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    // '®' to 'ÿ' 直接映射
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(unicode_cpt_to_utf8(ch)) == map.end()) {
            map[unicode_cpt_to_utf8(256 + n)] = ch;
            ++n;
        }
    }
    return map;
}

// 一个示例 bpe_words = {"Hi!", " ", "🙂" }
static std::vector<std::string> unicode_byte_encoding_process(const std::vector<std::string> & bpe_words) {
    std::vector<std::string> bpe_encoded_words;
    for (const auto & word : bpe_words) {
        std::string text_utf;
        // 先转换为码点，再转换回去
        auto utf_word =  unicode_cpts_from_utf8(word);
        for (size_t i = 0; i < utf_word.size(); ++i) {
            text_utf += unicode_cpt_to_utf8(utf_word[i]);
        }

        std::string encoded_token;
        for (char & c : text_utf) {
            // 这个的返回值是经过映射后的字节，有可能和它原本的字节不一致
            encoded_token += unicode_byte_to_utf8(c);
        }
        bpe_encoded_words.emplace_back(encoded_token);
    }
    return bpe_encoded_words;
}

// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static std::vector<size_t> unicode_regex_split_custom_gpt2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        // end 是当前片段的末尾位置
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // regex: 's|'t|'re|'ve|'m|'ll|'d
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = _get_cpt(pos+1);
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                // 判断单引号后面的那个字符是否是 r, v, l
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = _get_cpt(pos+2);
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            // regex: <space>?\p{L}+
            if (flags2.is_letter) {
                pos += (cpt == ' ');
                while (flags2.is_letter) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }
            // regex: <space>?\p{N}+
            if (flags2.is_number) {
                pos += (cpt == ' ');
                while (flags2.is_number) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }
            // regex: <space>?[^\s\p{L}\p{N}]+
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                num_whitespaces++;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// LLAMA3 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
static std::vector<size_t> unicode_regex_split_custom_llama3(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;  // _prev_end 永远指向“上一个已经剪好的片段”的末尾，也就是“下一个即将剪开的片段”的起始点
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?\p{L}+
            if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
                // 如果当前字符是字母，或者下一个字符是字母（比如空格）
                if (flags.is_letter || _get_flags(pos+1).is_letter) {  // one or more letters
                    pos++;
                    while (_get_flags(pos).is_letter) {
                        pos++;
                    }
                    // 可能是“空格+单词”，也可能是“纯单词”
                    _add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}{1,3}
            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3 ) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            // 如果这个字符既不是空格、也不是字母、也不是数字（那它就是标点、符号或特殊字符）。
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');  // 如果开头是空格，吃掉它
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);  // 如果符号后面跟着换行符，也吃掉
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // regex: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// Qwen2 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
static std::vector<size_t> unicode_regex_split_custom_qwen2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?\p{L}+
            if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
                if (flags.is_letter || _get_flags(pos+1).is_letter) {  // one or more letters
                    pos++;
                    while (_get_flags(pos).is_letter) {
                        pos++;
                    }
                    _add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}
            if (flags.is_number) {
                pos++;
                _add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // regex: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// Qwen3.5 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
// Compared to Qwen2, letter-runs also consume Unicode combining marks (\p{M}): [\p{L}\p{M}]+ instead of \p{L}+
static std::vector<size_t> unicode_regex_split_custom_qwen35(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
                if (flags.is_letter || flags.is_accent_mark || _get_flags(pos + 1).is_accent_mark || _get_flags(pos+1).is_letter) {
                    pos++;
                    while (_get_flags(pos).is_letter || _get_flags(pos).is_accent_mark) {
                        pos++;
                    }
                    _add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}
            if (flags.is_number) {
                pos++;
                _add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_accent_mark | flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_accent_mark | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // regex: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// 传入的是经过特殊字符替换后的文本，以及经过特殊字符替换后的正则表达式，这一步的 bpe_offsets 还是只有一个元素，其值为 cpts.size()
// 即使用 0xD1 等替换后的
template <typename CharT>
static std::vector<size_t> unicode_regex_split_stl(const std::basic_string<CharT> & text, const std::basic_string<CharT> & regex, const std::vector<size_t> & offsets) {
    // BidirIt 是 Bidirectional Iterator（双向迭代器）的缩写。这意味着这个“指针”可以向前走，也可以向后走
    using BidirIt = typename std::basic_string<CharT>::const_iterator;
#ifdef _MSC_VER
    // Bypass bug in MSVC: https://github.com/ggml-org/llama.cpp/issues/17830
    constexpr auto regex_flags = std::regex_constants::ECMAScript;
#else
    constexpr auto regex_flags = std::regex_constants::optimize | std::regex_constants::nosubs;
#endif
    std::basic_regex<CharT> expr(regex, regex_flags);  // 编译正则表达式
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;
    for (auto offset : offsets) {
        // std::regex_iterator 是 C++ 正则库里一个非常高级且好用的迭代器工具。它可以像遍历数组一样，去遍历文本里所有的正则表达式匹配项。
        // expr 是搜索规则
        std::regex_iterator<BidirIt> it(text.begin() + start, text.begin() + start + offset, expr);
        // 在 C++ 正则库里，一个 默认构造（即括号里什么都不传）的 regex_iterator 是一个特殊的信号。
        // 它代表“匹配结束”或者“没有更多匹配了”。
        std::regex_iterator<BidirIt> end;

        int64_t start_idx = 0;
        while (it != end) {
            std::match_results<BidirIt> match = *it;
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

// K2 system regex patterns (from tokenization_kimi.py):
// [\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
static std::vector<size_t> unicode_regex_split_custom_kimi_k2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // Pattern 1: [\p{Han}]+ (Chinese characters)
            if (unicode_cpt_is_han(cpt)) {
                while (unicode_cpt_is_han(_get_cpt(pos))) {
                    pos++;
                }
                _add_token(pos);
                continue;
            }

            // Pattern 2 & 3: Letter words excluding Han characters with optional contractions
            // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?:'s|'t|'re|'ve|'m|'ll|'d)?
            // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?:'s|'t|'re|'ve|'m|'ll|'d)?
            // Check if current char is a letter OR if current char could be a leading char and next char is a letter
            bool is_letter_pattern = (flags.is_letter && !unicode_cpt_is_han(cpt)) ||
                                     (!(cpt == '\r' || cpt == '\n' || flags.is_letter || flags.is_number) &&
                                      _get_flags(pos + 1).is_letter && !unicode_cpt_is_han(_get_cpt(pos + 1)));

            if (is_letter_pattern) {
                // Handle optional leading non-letter/non-number character
                bool has_leading_char = false;
                if (!(cpt == '\r' || cpt == '\n' || flags.is_letter || flags.is_number)) {
                    has_leading_char = true;
                    pos++;
                }

                // Match letter sequence (excluding Han characters)
                bool has_letters = false;
                while (_get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos))) {
                    has_letters = true;
                    pos++;
                }

                // Only proceed if we found letters (after potentially skipping leading char)
                if (has_letters || (!has_leading_char && _get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos)))) {
                    if (!has_letters) pos++; // consume the first letter if we didn't already

                    // Continue consuming letters
                    while (_get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos))) {
                        pos++;
                    }

                    // Check for optional contractions (?:'s|'t|'re|'ve|'m|'ll|'d)
                    if (_get_cpt(pos) == '\'' && pos + 1 < offset_end) {
                        // 取单引号后面的那个字符，并转为小写，不是就返回原字符
                        uint32_t cpt_next = unicode_tolower(_get_cpt(pos + 1));
                        // 判断单引号后面的那个字符是否是 s, t, m, d
                        if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                            pos += 2;
                        } else if (pos + 2 < offset_end) {
                            uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos + 2));
                            if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                                (cpt_next == 'v' && cpt_next_next == 'e') ||
                                (cpt_next == 'l' && cpt_next_next == 'l')) {
                                pos += 3;
                            }
                        }
                    }

                    _add_token(pos);
                    continue;
                } else if (has_leading_char) {
                    // We consumed a leading char but found no letters, backtrack
                    pos--;
                }
            }

            // Pattern 4: \p{N}{1,3} (numbers 1-3 digits)
            // regex: \p{N}{1,3} 三位一断
            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            // Pattern 5:  ?[^\s\p{L}\p{N}]+[\r\n]* (optional space + non-word chars + optional newlines)
            auto flags2 = (cpt == ' ' ? _get_flags(pos + 1) : flags);
            if (!(flags2.is_whitespace || flags2.is_letter || flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace || flags2.is_letter || flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                // Match optional [\r\n]*
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            // Count whitespace characters
            size_t num_whitespaces = 0;  // 记录从当前位置（pos）开始，连续有多少个空白字符（空格、换行、制表符等）。
            size_t last_end_r_or_n = 0;  // 记录这一串空白中，最后一个换行符（\r 或 \n） 所在的位置。
            while (_get_flags(pos + num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos + num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // Pattern 6: \s*[\r\n]+ (whitespace with newlines)
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // Pattern 7: \s+(?!\S) (trailing whitespace)
            if (num_whitespaces > 1 && _get_cpt(pos + num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // Pattern 8: \s+ (general whitespace)
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // No matches - consume single character
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// AFMOE digit handling: splits digits with leading 1-2 based on total length modulo 3
static std::vector<size_t> unicode_regex_split_custom_afmoe(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; ) {
            const auto flags = _get_flags(pos);

            // Handle digit sequences with special splitting logic
            if (flags.is_number) {
                size_t digit_start = pos;
                size_t digit_count = 0;

                // Count consecutive digits
                while (_get_flags(pos).is_number && pos < offset_end) {
                    digit_count++;
                    pos++;
                }

                // Split based on total length modulo 3
                size_t remainder = digit_count % 3;
                size_t current = digit_start;

                // Emit leading 1-2 digits if needed
                if (remainder > 0) {
                    _add_token(current + remainder);
                    current += remainder;
                }

                // Emit groups of 3
                while (current < digit_start + digit_count) {
                    _add_token(current + 3);
                    current += 3;
                }
                continue;
            }

            // For non-digits, just move forward
            pos++;
        }

        // Add any remaining content
        if (_prev_end < offset_end) {
            _add_token(offset_end);
        }
    }

    return bpe_offsets;
}

// regex: [^\n]+|[\n]+
// splits text into runs of non-newline characters and runs of newline characters
static std::vector<size_t> unicode_regex_split_custom_newlines(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        size_t pos = offset_ini;
        while (pos < offset_end) {
            const bool is_newline = (cpts[pos] == '\n');
            const size_t run_start = pos;
            while (pos < offset_end && (cpts[pos] == '\n') == is_newline) {
                pos++;
            }
            bpe_offsets.push_back(pos - run_start);
        }
    }

    return bpe_offsets;
}

// 传入的是 fragment.raw_text 和 unicode_regex_split 里面申明的 std::vector<size_t> bpe_offsets = { cpts.size() };即 offsets = bpe_offsets
static std::vector<size_t> unicode_regex_split_custom(const std::string & text, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;

    if (regex_expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
        bpe_offsets = unicode_regex_split_custom_gpt2(text, offsets);
    } else if (
            regex_expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
            regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        bpe_offsets = unicode_regex_split_custom_llama3(text, offsets);
    } else if (
           regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        bpe_offsets = unicode_regex_split_custom_qwen2(text, offsets);
    } else if (
           regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        bpe_offsets = unicode_regex_split_custom_qwen35(text, offsets);
    } else if (regex_expr == "\\p{Han}+") {
        // K2's first pattern - handle all K2 patterns together
        bpe_offsets = unicode_regex_split_custom_kimi_k2(text, offsets);
    } else if (regex_expr == "\\p{AFMoE_digits}") {
        // AFMOE digit pattern - use custom implementation for proper splitting
        bpe_offsets = unicode_regex_split_custom_afmoe(text, offsets);
    } else if (regex_expr == "[^\\n]+|[\\n]+") {
        bpe_offsets = unicode_regex_split_custom_newlines(text, offsets);
    } else if (regex_expr == "\\d{1,3}(?=(?:\\d{3})*\\b)") {
        // tiny_aya digit grouping pattern from tokenizer.json:
        //   {"type": "Split", "pattern": {"Regex": "\\d{1,3}(?=(?:\\d{3})*\\b)"}, "behavior": "Isolated"}
        // Splits digits into groups of 3 from the right (e.g., 1234567 -> 1, 234, 567)
        // TODO: Revisit this regex, in case there are any subtle tokenization differences with the original regex.
        bpe_offsets = unicode_regex_split_custom_afmoe(text, offsets);
    }

    return bpe_offsets;
}

//
// interface
//

// 把数字 “打包” 回标准字节流
// 比如将 中 0x4E2D 转换为 0xE4 0xB8 0xAD
std::string unicode_cpt_to_utf8(uint32_t cpt) {
    std::string result;

    if (/* 0x00 <= cpt && */ cpt <= 0x7f) {
        result.push_back(cpt);
        return result;
    }
    if (0x80 <= cpt && cpt <= 0x7ff) {
        result.push_back(0xc0 | ((cpt >> 6) & 0x1f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    if (0x800 <= cpt && cpt <= 0xffff) {
        result.push_back(0xe0 | ((cpt >> 12) & 0x0f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    if (0x10000 <= cpt && cpt <= 0x10ffff) {
        result.push_back(0xf0 | ((cpt >> 18) & 0x07));
        result.push_back(0x80 | ((cpt >> 12) & 0x3f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }

    throw std::invalid_argument("invalid codepoint");
}

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    auto comp = [] (const uint32_t cpt, const range_nfd & range) {
        return cpt < range.first;
    };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        auto it = std::upper_bound(unicode_ranges_nfd.begin(), unicode_ranges_nfd.end(), cpt, comp) - 1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}

// 传进来的实际上是 text，即 const auto cpts = unicode_cpts_from_utf8(text);
// 把一串混乱的“原始字节”，变成一排整齐的“Unicode 编号”
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        try {
            // 尝试从当前位置读取一个完整的 Unicode 字符
            result.push_back(unicode_cpt_from_utf8(utf8, offset));
        }
        catch (const std::invalid_argument & /*ex*/) {
            // Silently ignore invalid UTF-8 input to avoid leaking the exception beyond llama_tokenize
            ++offset;
            result.emplace_back(0xFFFD); // replacement character
        }
    }
    return result;
}

// 根据给出的编号（Codepoint），告诉这个字符到底是什么身份（是字母？数字？还是标点？）
unicode_cpt_flags unicode_cpt_flags_from_cpt(const uint32_t cpt) {
    // 如果查不到，默认返回“未定义”
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    // 静态初始化一张巨大的“身份表”，里面包含 14 多万个字符对应的身份信息。
    static const auto cpt_flags = unicode_cpt_flags_array();
    // 检查这个编号是否在表的范围内，如果在，就直接返回它对应的身份信息；如果超出了（比如你传入了一个更大的数字），就返回“未定义”。
    return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
}

unicode_cpt_flags unicode_cpt_flags_from_utf8(const std::string & utf8) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    if (utf8.empty()) {
        return undef;  // undefined
    }
    size_t offset = 0;
    return unicode_cpt_flags_from_cpt(unicode_cpt_from_utf8(utf8, offset));
}

std::string unicode_byte_to_utf8(uint8_t byte) {
    static std::unordered_map<uint8_t, std::string> map = unicode_byte_to_utf8_map();
    return map.at(byte);
}

uint8_t unicode_utf8_to_byte(const std::string & utf8) {
    static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    return map.at(utf8);
}

uint32_t unicode_tolower(uint32_t cpt) {
    // binary search
    auto it = std::lower_bound(unicode_map_lowercase.begin(), unicode_map_lowercase.end(), cpt,
        [](const std::pair<uint32_t, uint32_t> & pair, uint32_t value) {
            return pair.first < value;
        });
    if (it != unicode_map_lowercase.end() && it->first == cpt) {
        return it->second;
    }
    return cpt;  // Return the original code point if no lowercase mapping is found
}

bool unicode_cpt_is_han(uint32_t cpt) {
    // Han character ranges (Chinese/CJK characters)
    // CJK Unified Ideographs (most common)
    if (cpt >= 0x4E00 && cpt <= 0x9FFF) return true;

    // CJK Extension A
    if (cpt >= 0x3400 && cpt <= 0x4DBF) return true;

    // CJK Extension B
    if (cpt >= 0x20000 && cpt <= 0x2A6DF) return true;

    // CJK Extension C
    if (cpt >= 0x2A700 && cpt <= 0x2B73F) return true;

    // CJK Extension D
    if (cpt >= 0x2B740 && cpt <= 0x2B81F) return true;

    // CJK Extension E
    if (cpt >= 0x2B820 && cpt <= 0x2CEAF) return true;

    // CJK Extension F
    if (cpt >= 0x2CEB0 && cpt <= 0x2EBEF) return true;

    // CJK Compatibility Ideographs
    if (cpt >= 0xF900 && cpt <= 0xFAFF) return true;

    // CJK Compatibility Ideographs Supplement
    if (cpt >= 0x2F800 && cpt <= 0x2FA1F) return true;

    return false;
}

// 传入的是 fragment.raw_text
std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs, bool byte_encode) {
    // unicode categories
    static const std::map<std::string, int> k_ucat_enum = {
        { "\\p{N}", unicode_cpt_flags::NUMBER },  // 所有数字，含罗马数字、分数等
        { "\\p{L}", unicode_cpt_flags::LETTER },  // 所有语言的字母
        { "\\p{P}", unicode_cpt_flags::PUNCTUATION },  // 所有的标点符号
        { "\\p{M}", unicode_cpt_flags::ACCENT_MARK },  // 所有的变音符号
        { "\\p{S}", unicode_cpt_flags::SYMBOL },  // 数学符号、货币符号、Emoji
        { "\\p{Lu}", unicode_cpt_flags::LETTER },  // 大写字母
        { "\\p{Ll}", unicode_cpt_flags::LETTER },  // 小写字母
        { "\\p{Lt}", unicode_cpt_flags::LETTER },  // 首字母大写字母
        { "\\p{Lm}", unicode_cpt_flags::LETTER },  // 修饰字母
        { "\\p{Lo}", unicode_cpt_flags::LETTER },  // 其他字母
    };

    static const std::map<int, int> k_ucat_cpt = {
        { unicode_cpt_flags::NUMBER,      0xD1 },  // 数字
        { unicode_cpt_flags::LETTER,      0xD2 },  // 字母
        { unicode_cpt_flags::PUNCTUATION, 0xD3 },  // 标点符号
        { unicode_cpt_flags::ACCENT_MARK, 0xD4 },  // 变音符号
        { unicode_cpt_flags::SYMBOL,      0xD5 },  // 符号
    };

    static const std::map<int, std::string> k_ucat_map = {
        { unicode_cpt_flags::NUMBER,      "\x30-\x39" }, // 0-9
        { unicode_cpt_flags::LETTER,      "\x41-\x5A\x61-\x7A" }, // A-Za-z
        { unicode_cpt_flags::PUNCTUATION, "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D" }, // !-#%-*,-/:-;?-@\[-\]_\{\}
        { unicode_cpt_flags::ACCENT_MARK, "" }, // no sub-128 codepoints
        { unicode_cpt_flags::SYMBOL,      "\\\x24\\\x2B\x3C-\x3E\x5E\x60\\\x7C\\\x7E" }, // $+<=>^`|~
    };

    // compute collapsed codepoints only if needed by at least one regex
    bool need_collapse = false;
    for (const auto & regex_expr : regex_exprs) {
        // search for unicode categories
        for (const auto & ucat : k_ucat_enum) {
            if (std::string::npos != regex_expr.find(ucat.first)) {
                need_collapse = true;
                break;
            }
        }
    }

    // 输入：是一串原始的字节，像 0xC3 0xA9 这样零散的数据。
    // 输出：是一串数字 codepoints，像 233 这样代表 Unicode 字符的编号。
    const auto cpts = unicode_cpts_from_utf8(text);

    // generate a "collapsed" representation of the text, where all codepoints are replaced by a single byte
    // ref: https://github.com/ggml-org/llama.cpp/pull/6920#issuecomment-2081479935
    std::string text_collapsed;
    if (need_collapse) {
        // collapse all unicode categories
        text_collapsed.resize(cpts.size());

        for (size_t i = 0; i < cpts.size(); ++i) {
            // keep single-byte codepoints as is
            // keep single-byte codepoints as is
            // 128 以下也就是 ASCII 字符，对于 C++ 的正则引擎（std::regex）来说，它本生就是为了处理单字节的 ASCII 字符设计的。
            // 它认识 A-Z。它认识 0-9。它认识空格、标点符号。
            // 所以，根本没必要把一个本来就是单字节的 A 再映射成另一个单字节的 X。直接让它保持原样，正则引擎就能处理得很好。
            // 当代码点 大于等于 128（即非 ASCII 字符，比如中文、韩文、带重音的法语）时，映射才真正开始工作
            if (cpts[i] < 128) {
                text_collapsed[i] = cpts[i];
                continue;
            }

            const auto flags = unicode_cpt_flags_from_cpt(cpts[i]);

            if (flags.is_whitespace) {
                //NOTE: C++ std::regex \s does not mach 0x85, Rust and Python regex does.
                //text_collapsed[i] = (char) 0x85;  // <Next Line> as whitespace fallback
                //NOTE: C++ std::regex \s does not mach 0x85, Rust and Python regex does.
                //text_collapsed[i] = (char) 0x85;  // <Next Line> as whitespace fallback
                // 不管是哪国空格，只要是空格，就统一替换成 ASCII 码里的 0x0B（垂直制表符）
                text_collapsed[i] = (char) 0x0B;    // <vertical tab> as whitespace fallback
            } else if (k_ucat_cpt.find(flags.category_flag()) != k_ucat_cpt.end()) {
                // 它查了一张映射表（k_ucat_cpt），把复杂的字符变成了一个简单的字节
                // 如果是字母 (Letter)：不论是中文“你”、泰文还是德语，统一变成字节 0xD2。
                // 如果是数字 (Number)：统一变成字节 0xD1。
                // 如果是标点 (Punctuation)：统一变成字节 0xD3。
                text_collapsed[i] = k_ucat_cpt.at(flags.category_flag());
            } else {
                text_collapsed[i] = (char) 0xD0; // fallback
            }
        }
    }

    // 初始化一个只有一个元素的 vector，而这个元素的值是 cpts.size()。
    std::vector<size_t> bpe_offsets = { cpts.size() };

    // 分词逻辑
    for (const auto & regex_expr : regex_exprs) {
        // first, see if we have an efficient custom regex implementation
        // first, see if we have an efficient custom regex implementation
        // 这个函数返回的是每个 token 的长度，举个例子 bpe_offsets = {8, 12, 5, 11, 13, 1}。
        auto tmp = unicode_regex_split_custom(text, regex_expr, bpe_offsets);

        if (!tmp.empty()) {
            bpe_offsets = std::move(tmp);
            continue;
        }

        // fallback to general-purpose std::regex / std::wregex
        try {
            // if a unicode category is used in the regex, we use the collapsed text and replace the unicode category
            // with the corresponding collapsed representation
            bool use_collapsed = false;
            for (const auto & ucat : k_ucat_enum) {
                if (std::string::npos != regex_expr.find(ucat.first)) {
                    use_collapsed = true;
                    break;
                }
            }
            // sanity-check that the original regex does not contain any non-ASCII characters
            // 不包含泛化 Unicode 属性（如 \p{L}）的简单正则表达式
            // 把正则表达式转成“以字符为单位”的序列，这样匹配引擎就能按“字符”匹配，而不会把字符切碎。
            // 把 “长短不一的一串字节” 翻译成 “一个萝卜一个坑的一串字符”
            const auto cpts_regex = unicode_cpts_from_utf8(regex_expr);

            if (use_collapsed) {
                // sanity-check that the original regex does not contain any non-ASCII characters
                for (size_t i = 0; i < cpts_regex.size(); ++i) {
                    if (cpts_regex[i] >= 128) {
                        throw std::runtime_error("Regex includes both unicode categories and non-ASCII characters - not supported");
                    }
                }

                // generate a collapsed representation of the regex
                std::string regex_expr_collapsed;

                // track if we are inside [], because nested [] are not allowed
                bool inside = false;  // 记录现在是不是正处于正则表达式的中括号 [...] 内部
                for (size_t i = 0; i < regex_expr.size(); ++i) {
                    // '\\'代表的其实是字符'\'
                    if (regex_expr[i] == '[' && (i == 0 || regex_expr[i - 1] != '\\')) {
                        regex_expr_collapsed += '[';
                        inside = true;
                        continue;
                    }

                    if (inside && regex_expr[i] == ']' && regex_expr[i - 1] != '\\') {
                        regex_expr_collapsed += ']';
                        inside = false;
                        continue;
                    }

                    // Match \p{...} Unicode properties of varying lengths
                    if (regex_expr[i + 0] == '\\' && i + 3 < regex_expr.size() &&
                        regex_expr[i + 1] == 'p' &&
                        regex_expr[i + 2] == '{') {
                        // Find the closing brace
                        size_t closing_brace = regex_expr.find('}', i + 3);
                        if (closing_brace != std::string::npos && closing_brace <= i + 10) { // reasonable limit
                            const std::string pat = regex_expr.substr(i, closing_brace - i + 1);
                            if (k_ucat_enum.find(pat) != k_ucat_enum.end()) {
                                if (!inside) {
                                    regex_expr_collapsed += '[';
                                }
                                // 比如 pat 是 \p{L}，那么 regex_expr_collapsed 会存入[\xD2A-Za-z]
                                regex_expr_collapsed += k_ucat_cpt.at(k_ucat_enum.at(pat));
                                regex_expr_collapsed += k_ucat_map.at(k_ucat_enum.at(pat));
                                if (!inside) {
                                    regex_expr_collapsed += ']';
                                }
                                i = closing_brace;
                                continue;
                            }
                        }
                    }

                    regex_expr_collapsed += regex_expr[i];
                }

                //printf("text_collapsed: %s\n", text_collapsed.c_str());
                //printf("regex_expr_collapsed: %s\n", regex_expr_collapsed.c_str());
                bpe_offsets = unicode_regex_split_stl(text_collapsed, regex_expr_collapsed, bpe_offsets);
            } else {
                // no unicode category used, we can use std::wregex directly
                std::wstring wregex_expr(cpts_regex.begin(), cpts_regex.end());

                // std::wregex \s does not mach non-ASCII whitespaces, using 0x0B as fallback
                std::wstring wtext(cpts.begin(), cpts.end());
                for (size_t i = 0; i < wtext.size(); ++i) {
                    // 如果这个字符不是 ASCII 字符 (> 0x7F) 并且根据我们自己的 Unicode 旗帜表，它被标记为“空白符” (is_whitespace)
                    if (wtext[i] > 0x7F && unicode_cpt_flags_from_cpt(wtext[i]).is_whitespace) {
                        wtext[i] = 0x0B;
                    }
                }

                //printf("text: %s\n", text.c_str());
                //printf("regex_expr: %s\n", regex_expr.c_str());
                bpe_offsets = unicode_regex_split_stl(wtext, wregex_expr, bpe_offsets);
            }
        } catch (std::regex_error & e) {
            fprintf(stderr, "Failed to process regex: '%s'\n", regex_expr.c_str());
            fprintf(stderr, "Regex error: %s\n", e.what());
            throw std::runtime_error("Failed to process regex");
        }
    }

    // 存放这些最终词块的仓库，假设之前的数字是 {2, 1, 5}，接下来的目标是把它们变成具体的："Hi", " ", "World"。
    std::vector<std::string> bpe_words;
    bpe_words.reserve(bpe_offsets.size()); // reserve memory for the approximate size

    size_t start = 0;
    for (size_t & offset : bpe_offsets) {
        bpe_words.emplace_back();
        for (size_t i = start; i < start + offset; ++i) {
            // 把 32 位的数字 (如 0x1F642) 转回多字节的 UTF-8 字符串，然后累加进当前这个单词里
            bpe_words.back() += unicode_cpt_to_utf8(cpts[i]);
        }
        start += offset;
    }

    if (byte_encode) {
        // 至此，bpe_words 里面装的就是“切好的小词块”，一个示例 bpe_words = {"Hi!", " ", "🙂" }
        return unicode_byte_encoding_process(bpe_words);
    }

    return bpe_words;
}
