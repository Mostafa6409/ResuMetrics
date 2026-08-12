from unstructured.partition.pdf import partition_pdf


def extract(cv):
   elements = partition_pdf(filename=cv, strategy="auto")
   resume_text = "\n\n".join([str(e) for e in elements])

   return resume_text

