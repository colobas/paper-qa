import warnings
from pathlib import Path
from typing import List

from html2text import html2text
from langchain.text_splitter import TokenTextSplitter

from .types import Doc, Text


def parse_pdf_fitz(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
    import fitz

    file = fitz.open(path)
    split = ""
    pages: List[str] = []
    texts: List[Text] = []
    for i in range(file.page_count):
        page = file.load_page(i)
        split += page.get_text("text", sort=True)
        pages.append(str(i + 1))
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            texts.append(
                Text(
                    text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc
                )
            )
            split = split[chunk_chars - overlap :]
            pages = [str(i + 1)]
    if len(split) > overlap:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc)
        )
    file.close()
    return texts


def parse_pdf_nougat(
    path: Path, doc: Doc, chunk_chars: int, overlap: int
) -> List[Text]:
    import re
    from functools import partial

    import torch
    from nougat.model import NougatModel
    from nougat.utils.checkpoint import get_checkpoint
    from nougat.utils.dataset import LazyDataset

    if torch.cuda.is_available():
        _use_cuda = True
        _batchsize = int(
            torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1000 * 0.3
        )
        if _batchsize == 0:
            warnings.warn(
                "Your GPU does not have enough memory to run nougat. "
                "Will use CPU, which will be slow."
            )
            _batchsize = 5
            _use_cuda = False
    else:
        _use_cuda = False
        warnings.warn("To use `nougat` you should use a GPU.")
        _batchsize = 5  # default from nougat repo

    _checkpoint = None
    _pdf = path
    if _checkpoint is None or not _checkpoint.exists():
        _checkpoint = get_checkpoint(_checkpoint)

    model = NougatModel.from_pretrained(_checkpoint).to(torch.bfloat16)
    if _use_cuda:
        model = model.cuda()

    dataset = LazyDataset(
        _pdf, partial(model.encoder.prepare_input, random_padding=False)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=_batchsize,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )

    pages: List[str] = []
    texts: List[Text] = []

    page_num = 0

    split = ""
    for i, (sample, is_last_page) in enumerate(dataloader):
        model_output = model.inference(image_tensors=sample)
        # check if model output is faulty
        for j, output in enumerate(model_output["predictions"]):
            page_num += 1
            pages.append(str(page_num))
            if output.strip() == "[MISSING_PAGE_POST]":
                # uncaught repetitions -- most likely empty page
                output = f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n"
            elif model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    # If we end up here, it means the output is most likely not complete and was truncated.
                    output = f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n"
                else:
                    # If we end up here, it means the document page is too different from the training domain.
                    # This can happen e.g. for cover pages.
                    output = f"\n\n[MISSING_PAGE_EMPTY:{i*_batchsize+j+1}]\n\n"

            output = re.sub(r"\n{3,}", "\n\n", output).strip()
            split += output

            while len(split) > chunk_chars:
                # pretty formatting of pages (e.g. 1-3, 4, 5-7)
                pg = "-".join([pages[0], pages[-1]])
                texts.append(
                    Text(
                        text=split[:chunk_chars],
                        name=f"{doc.docname} pages {pg}",
                        doc=doc,
                    )
                )
                split = split[chunk_chars - overlap :]
                pages = [str(i + 1)]
    if len(split) > overlap:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc)
        )
    return texts


def parse_pdf(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
    import pypdf

    pdfFileObj = open(path, "rb")
    pdfReader = pypdf.PdfReader(pdfFileObj)
    split = ""
    pages: List[str] = []
    texts: List[Text] = []
    for i, page in enumerate(pdfReader.pages):
        split += page.extract_text()
        pages.append(str(i + 1))
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            texts.append(
                Text(
                    text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc
                )
            )
            split = split[chunk_chars - overlap :]
            pages = [str(i + 1)]
    if len(split) > overlap:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc)
        )
    pdfFileObj.close()
    return texts


def parse_txt(
    path: Path, doc: Doc, chunk_chars: int, overlap: int, html: bool = False
) -> List[Text]:
    try:
        with open(path) as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    if html:
        text = html2text(text)
    # yo, no idea why but the texts are not split correctly
    text_splitter = TokenTextSplitter(chunk_size=chunk_chars, chunk_overlap=overlap)
    raw_texts = text_splitter.split_text(text)
    texts = [
        Text(text=t, name=f"{doc.docname} chunk {i}", doc=doc)
        for i, t in enumerate(raw_texts)
    ]
    return texts


def parse_code_txt(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
    """Parse a document into chunks, based on line numbers (for code)."""

    split = ""
    texts: List[Text] = []
    last_line = 0

    with open(path) as f:
        for i, line in enumerate(f):
            split += line
            if len(split) > chunk_chars:
                texts.append(
                    Text(
                        text=split[:chunk_chars],
                        name=f"{doc.docname} lines {last_line}-{i}",
                        doc=doc,
                    )
                )
                split = split[chunk_chars - overlap :]
                last_line = i
    if len(split) > overlap:
        texts.append(
            Text(
                text=split[:chunk_chars],
                name=f"{doc.docname} lines {last_line}-{i}",
                doc=doc,
            )
        )
    return texts


def read_doc(
    path: Path,
    doc: Doc,
    chunk_chars: int = 3000,
    overlap: int = 100,
    force_pypdf: bool = False,
    force_nougat: bool = False,
) -> List[Text]:
    """Parse a document into chunks."""
    str_path = str(path)
    if str_path.endswith(".pdf"):
        if force_pypdf:
            return parse_pdf(path, doc, chunk_chars, overlap)
        if force_nougat:
            return parse_pdf_nougat(path, doc, chunk_chars, overlap)
        try:
            return parse_pdf_fitz(path, doc, chunk_chars, overlap)
        except ImportError:
            return parse_pdf(path, doc, chunk_chars, overlap)
    elif str_path.endswith(".txt"):
        return parse_txt(path, doc, chunk_chars, overlap)
    elif str_path.endswith(".html"):
        return parse_txt(path, doc, chunk_chars, overlap, html=True)
    else:
        return parse_code_txt(path, doc, chunk_chars, overlap)
